import time
from datetime import timedelta
import pandas as pd
from pathlib import Path
from pprint import pformat
import os, sys
import re
import psycopg2 as pg2
import traceback
import pylogpresso
from pylogpresso.client import StreamingResultSet
import algorithms.logseq.config.default_regex as basereg
from algorithms.ocdigcn.ocdigcn import OCDiGCN
from analyzer import aimodule
from api.tsa.tsa_utils import TSAUtils
from common.clickhouse_client import get_client, close_client
from common import aicommon, constants
from common.error_code import Errors
from common.module_exception import ModuleException
from common.base64_util import Base64Util
from resources.config_manager import Config

class ExemAiopsAnlsLog(aimodule.AIModule):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        config_path = os.environ.get("MLOPS_TRAINING_PATH")
        if config_path is None:
            print("plz export MLOPS_TRAINING_PATH", file=sys.stderr, flush=True)
            sys.exit()
        else:
            self.logpresso_config = Config(config_path, os.environ.get("AIMODULE_SERVER_ENV")).get_config()
        self.logpresso_decode_config = Base64Util.get_config_decode_value(self.logpresso_config["logpresso"])

        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.service_id = f"{self.module_id}_{config['target_id']}"

        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]
        config["model_dir"] = str(Path(self.config["home"]) / "model" / str(self.sys_id) / config["module"] / "log" / self.target_id)
        self.inst_id = config.get("inst_id", 0)
        if self.inst_type == constants.INST_TYPE_OS:
            self.inst_id = 0
        self.preset_list = []
        self.delimiters = " "
        self.log_length = 300
        self.logger.info(f"config :{pformat(config)}")

        # MLOps TSA
        self.tsa = TSAUtils()

        try:
            self.digcn = OCDiGCN(config=config, logger=logger)
        except Exception as exception:
            logger.exception(f"[Error] Unexpected exception during instance : {exception}")
            raise ModuleException("E709")

    def _save(self):
        model_dir = self.config["model_dir"]
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        self.digcn.save(model_dir)

    def _get_preset_type_from_pg(self, target_id):
        conn, cursor = None, None
        regex_list, delimiters = [], " "
        try:
            conn = pg2.connect(self.config["db_conn_str"])
            cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

            query = "select re.regset_id, re.delimiter, re.regex, re.replace_str " \
                    "from ai_config_log_regex re left join xaiops_config_log xl " \
                    "on re.regset_id = xl.regset_id " \
                    "where xl.target_id = %s"
            cursor.execute(query, (target_id,))
            preset_result = [dict(record) for record in cursor]
            if len(preset_result) > 0:
                for item in preset_result:
                    if item["delimiter"]:
                        delimiters = item["regex"]
                    else:
                        regex = [item["regex"], item["replace_str"]]
                        regex_list.append(regex)
            else:
                regex_list = basereg.COMMON_REGEX
                delimiters = basereg.COMMON_DELIMITER
            self.preset_list = regex_list
            self.delimiters = delimiters

            query_category = "select log_type " \
                            "from xaiops_config_log " \
                            "where target_id = %s"
            cursor.execute(query_category, (target_id,))
            category_result = cursor.fetchone()
            # log_type은 syslog 또는 filelog 중 하나이며 이에 따라 학습 데이터 조회 할 dm테이블명이 달라짐
            log_type = category_result['log_type']

        except Exception as ex:
            self.logger.exception(f"unable to get regex from the database : {ex}\n\n{traceback.format_exc()}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return log_type

    def query_train_data_from_ch(self, log_type):
        line_limit = self.config["parameter"]["train"][constants.MODEL_S_DIGCN]["metric_hyper_params"]["msg"]["line_limit"]
        query_result = []
        start_time = time.time()
        day_count = 0
        try:
            client = get_client()
            for i in range(1, len(self.config["date"])+1):
                day_result = []
                start, end = self.config["date"][-i], (pd.to_datetime(self.config["date"][-i]) + timedelta(days=1)).strftime("%Y%m%d")
                query_str = f"""select time, left(message, {self.log_length})
                                from dm_{log_type}
                                where target_id = '{self.target_id}' and DATE(time) BETWEEN '{start}' AND '{end}'
                            """
                with client.query_row_block_stream(query_str) as stream:
                    for block in stream:
                        day_result.extend(block)
                remain_limit = line_limit - len(query_result)

                day_count += 1
                if len(day_result) > remain_limit:
                    if day_count < 10:
                        query_result = day_result + query_result
                    else:
                        query_result = day_result[-remain_limit:] + query_result
                        break
                else:
                    query_result = day_result + query_result
                self.logger.info(f"{start} ~ {end}, day_result: {len(day_result)}, result sum: {len(query_result)}, remain_limit: {line_limit - len(query_result)}")

        except Exception as ex:
            self.logger.exception(f"error occurred in Logpresso query : {ex}\nreconnecting Logpresso client...")
        if client:
            close_client()
        result_df = pd.DataFrame(query_result, columns=['time', 'msg'])
        self.logger.info(f"*** total day count: {day_count}, elapsed time: {time.time() - start_time}s | total data length: {len(result_df)}")
        return result_df

    def replace_preset_and_remove_number(self, df):
        start_time = time.time()
        self.logger.info(f"original msg >>>>> \n{df['msg']}")
        if self.preset_list:
            self.logger.info(f"target_id : {self.target_id}, preset_list: {self.preset_list}")
            for preset in self.preset_list:
                try:
                    df['msg'] = list(map(lambda x: re.sub(preset[0], preset[1], x), df['msg']))
                except Exception as ex:
                    self.logger.exception(f"exception occurred in regex replace. preset: {preset} : {ex}\n\n{traceback.format_exc()}")
            self.logger.info(f"===== after preset replacing >>>>>  \n{df['msg']}")
        df['msg'] = list(map(lambda x: re.sub(r'[^\w\s]', ' ', x), df['msg']))   # remain only word, ...
        df['msg'] = list(map(lambda x: re.sub(r'\d', '', x), df['msg']))   # remove digit
        self.logger.info(f"===== after remove >>>>> \n{df['msg']}")
        self.logger.info(f"replace_preset_and_remove_number elapsed time: {time.time() - start_time}")
        return df

    def train(self, stat_logger):
        self.logger.info(f"exem_aiops_anls_log, train(), start..")

        result = dict()
        process_start_time = time.time()
        train_prog = {constants.PROCESS_TRAIN_DATA: self.default_module_status()}
        train_prog[constants.MODEL_B_DIGCN] = self.default_module_status()

        log_type = self._get_preset_type_from_pg(self.target_id)
        self.tsa.update_train_status(self.config['train_history_id'], 3)
        df = self.query_train_data_from_ch(log_type)
        self.tsa.update_train_status(self.config['train_history_id'], 4)

        if df is None or len(df) == 0:
            self.logger.info("[training] no train data")
            return None, None, Errors.E800.value, Errors.E800.desc
        elif df.shape[0] < 1000:  # 1000 => temp limit
            self.logger.info(f"[training] not enough training data")
            return None, None, Errors.E801.value, Errors.E801.desc

        aicommon.Query.update_module_status_by_training_id(
            self.config['db_conn_str'],
            self.config['train_history_id'],
            train_prog,
            train_process=constants.PROCESS_TRAIN_DATA,
            progress=100,
            process_start_time=f"{time.time() - process_start_time:.3f}"
        )

        df = self.replace_preset_and_remove_number(df)

        tmp_df = df.copy()
        digcn_result = self.digcn.fit(tmp_df, train_progress=train_prog)
        result[constants.MODEL_S_DIGCN] = digcn_result
        result['log_type'] = log_type

        self._save()
        self.logger.info(f"train_result : {result}")

        return result, None, 0, None

    def end_train(self):
        pass

