import json
import os
import time
import datetime
import numpy as np
import pandas as pd
import psycopg2 as pg2
import psycopg2.extras

from pathlib import Path
from copy import deepcopy
from filelock import FileLock
from analyzer import aimodule
from dateutil.parser import parse
from common.error_code import Errors
from common import aicommon, constants
from sklearn.preprocessing import StandardScaler
from algorithms.dynamic_baseline import DynamicBaseline
from common.module_exception import ModuleException
from common.redisai import REDISAI


def download_model_from_DB(rsync_model_to_db, model_path, logger):
    rsync_model_to_db.is_model_dir()  # 모델 디렉토리, 로그 디렉토리가 없으면 생성
    try:
        if not rsync_model_to_db.model_lock_file.exists():
            rsync_model_to_db.model_lock_file.touch()  # 모델 락을 만든다.
        with FileLock(rsync_model_to_db.model_lock_file):
            logger.info('test')
            with FileSplitMerge(model_path=model_path, logger=logger, max_file_size=200 * MB_SIZE):
                rsync_model_to_db.load_pg_to_path()  # 모델 다운로드
    except Exception as e:
        logger.exception(f"Unexpected Exception {e} occured while downloading model")


class ExemAiopsFutureFcst(aimodule.AIModule):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.inst_type = config['inst_type']
        self.module = config['module']
        self.sys_id = config['sys_id']
        self.target_id = config['target_id']

        self.from_date = str(config['from_date'])
        self.to_date = str(config['to_date'])

        self.future_features = config['collect_metric']

        if self.config['algorithm'] == 'seq2seq':
            self.algorithm = config['algorithm']
            self.model_name = 's2s_attn'
        else:
            self.algorithm = config['algorithm']
            self.model_name = self.algorithm
        self.db_conn_str = config['db_conn_str']

        self.master_id = None

        #예측으로 사용하는 데이터 상태 : min, max, std, avg
        # self.real_value_state = 'avg'

        self.use_dbsln = None
        self.loaded_model = None
        self.dbsln_model_dir = None
        self.load_fcst_model_dir = None
        self.dbsln_model_config = None
        self.loadfcst_model_config = None

        self.models = dict()

        self.init_model_config()
        self.business_list = self.config['business_list']
        self.model_load() # self.models[feat]에 model key 값 저장

        self.logger.info('model_config and data load complete!!')

    def init_model_config(self):
        home = os.environ.get("AIMODULE_HOME")

        required_model_for_analysis = "exem_aiops_anls_inst"
        self.service_id = f"{required_model_for_analysis}_{self.target_id}"
        self.dbsln_model_dir = str(
            Path(home) / "model" / str(self.sys_id) / required_model_for_analysis / self.inst_type / str(
                self.target_id)) + "/"

        if self.inst_type == "db":
            target_id = self.config.get("inst_product_type", "all")
        else:
            target_id = 'all'

        self.load_fcst_model_dir = str(
            Path(home) / "model" / str(self.sys_id) / constants.EXEM_AIOPS_LOAD_FCST / self.inst_type / target_id) + "/"

        try:
            dbsln_model_config_path = self.dbsln_model_dir + "model_config.json"
            dbsln_model_config_key = REDISAI.make_redis_model_key(dbsln_model_config_path, ".json")
            self.dbsln_model_config = REDISAI.inference_json(dbsln_model_config_key)

            loadfcst_model_config_path = self.load_fcst_model_dir + "model_config.json"
            loadfcst_model_config_key = REDISAI.make_redis_model_key(loadfcst_model_config_path, ".json")
            self.loadfcst_model_config = REDISAI.inference_json(loadfcst_model_config_key)
        except Exception as e:
            self.logger.exception(e)
            raise ModuleException("E850")

        self.dbsln_model_config['collect_metric'] = self.future_features


    def model_load(self):

        try:
            self.loaded_model_dbsln = DynamicBaseline(self.service_id, self.dbsln_model_config, self.logger)
            self.loaded_model_dbsln.load(self.dbsln_model_dir)
        except MemoryError as error:
            self.logger.exception('[Error] Unexpected memory error during serving... %s', error)
            aicommon.Utils.print_memory_usage(self.logger)
            raise error
        except Exception as exception:
            self.logger.exception('[Error] Unexpected exception during serving... %s', exception)
            raise exception

        for feature in self.future_features:
            onnx_feat_model_key = str(Path(self.load_fcst_model_dir) / f"s2s_attn_{feature}")
            model_key = REDISAI.make_redis_model_key(onnx_feat_model_key, "")
            self.models[feature] = model_key


    def set_timestamp(self, data, feat):
        res = []
        window_size = self.loadfcst_model_config['parameter']['train'][self.algorithm]['metric_hyper_params'][feat].get('window_size',30)
        for i in range(window_size, len(data)):
            res.append(data[i - window_size:i])
        res = np.array(res)
        res = np.reshape(res, (res.shape[0], res.shape[1], 1))
        return res

    def predict_band_merge(self, predict_data, band):
        band.reset_index(inplace=True)
        temp = band.copy()
        band['time'] = temp['time'].map(lambda x: str(parse(str(x))))

        results = dict()
        for k, v in predict_data.items():
            values = list()
            for i in zip(v, band[f"{k}_upper"], band[f"{k}_lower"]):
                values.append(i)
            results[k] = deepcopy(values)

        res = {"header": ['time', 'value', 'upper', 'lower', 'anomaly']}
        res_total = dict()
        for k, v in results.items():
            val_list = list()
            for i in range(len(v)):
                f_vals = list(map(float, v[i]))
                val_list.append([band['time'][i]] + f_vals)
                if f_vals[0] > f_vals[1] or f_vals[0] < f_vals[2]:
                    val_list[i].append(True)
                else:
                    val_list[i].append(False)

            res['body'] = deepcopy(val_list)
            res_total[k] = deepcopy(res)

        return res_total

    def train(self, train_logger):
        df_list = []
        for filename in os.listdir(self.config['train_dir']):
            if filename.endswith('.csv'):
                df = self.load_csvfiles([self.config['train_dir'] + '/' + filename])
            else:
                continue

            df = df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)
            df_list.append(df)
        df_list = pd.concat(df_list)

        self.logger.info(f"training data load done")

        result = aicommon.Utils.validate_trainig_dataframe(df_list, 120, self.logger)
        if type(result) == dict:
            return None, {"results": False, "master_id": False}, result["code"], result["message"]
        else:
            df_list = result

        self.logger.info(f"predict start")

        # 임시 조치
        df_list.sort_values(by=['time'], inplace=True)
        df_list.reset_index(drop=True, inplace=True)

        temp = df_list.loc[51:]
        res = dict()
        scaler = StandardScaler()
        for feat in self.future_features:
            data = temp[feat].values.reshape(-1, 1).astype('float32')
            scalered = scaler.fit(data)
            pred_data = scalered.transform(data)

            pred_data = self.set_timestamp(pred_data, feat)
            model_key = self.models[feat]
            pred_res = REDISAI.inference(model_key, pred_data)[0]
            pred_res = scalered.inverse_transform(np.squeeze(pred_res))

            inverse_transform_data = list()
            for i in range(len(pred_res)):
                inverse_transform_data.append(pred_res[i][0])

            inverse_transform_data = np.array(inverse_transform_data)
            res[feat] = np.where(inverse_transform_data < 0, 0, inverse_transform_data)

        dbsln_res = self.loaded_model_dbsln.predict([parse(self.from_date)+datetime.timedelta(minutes=x) for x in range(int((parse(self.to_date)-parse(self.from_date)).total_seconds()/60))], pd.DataFrame([]), mode="future")

        merge_result = self.predict_band_merge(res, dbsln_res)
        self.logger.info("finish make upper, lower !!")

        self.logger.info("predict for each features finish!")
        self.save(merge_result)

        return None, {"results": True, "master_id": self.master_id}, 0, None

    def save(self, results):

        def save_master():
            create_at = int(time.time())
            conn = None
            cursor = None
            try:
                conn = pg2.connect(self.db_conn_str)
                cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
                query = (
                    "with rows as( "
                    "INSERT INTO xaiops_module_future_fcst_master "
                    '("sys_id", "inst_type", "target_id", "algorithm", "from_date", "to_date", "create_date") '
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING master_id "
                    ") select master_id from rows "
                )

                cursor.execute(
                    query,
                    (self.sys_id, self.inst_type, self.target_id, self.algorithm, self.from_date, self.to_date, datetime.datetime.fromtimestamp(create_at).strftime(constants.INPUT_DATETIME_FORMAT)
                    ),
                )
                rows = cursor.fetchone()
                self.master_id = rows[0]
                conn.commit()
            except Exception as ex:
                self.logger.info(ex)
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            self.logger.info("master value save!")


            #     conn = pg2.connect(self.db_conn_str)
            #     cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            #     query = (
            #         "INSERT INTO xaiops_module_future_fcst_master "
            #         '("master_id",  "sys_id", "inst_type", "target_id", "algorithm", "from_date", "to_date", "create_date") '
            #         "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            #     )
            #
            #     cursor.execute(
            #         query,
            #         (self.master_id, self.sys_id, self.inst_type, self.target_id, self.algorithm, self.from_date, self.to_date, datetime.datetime.fromtimestamp(create_at).strftime(constants.INPUT_DATETIME_FORMAT)
            #         ),
            #     )
            #     conn.commit()
            # except Exception as ex:
            #     self.logger.info(ex)
            # finally:
            #     if cursor:
            #         cursor.close()
            #     if conn:
            #         conn.close()
            #
            # self.logger.info("master value save!")

        def save_predict():
            conn = None
            cursor = None
            try:
                conn = pg2.connect(self.db_conn_str)
                cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
                for feature, v in results.items():
                    query = (
                        "INSERT INTO xaiops_module_future_fcst_predict "
                        '("master_id", "metric", "predict_values") '
                        "VALUES (%s, %s, %s)"
                    )
                    cursor.execute(query,(self.master_id, feature, json.dumps(v)))
                conn.commit()
            except Exception as ex:
                self.logger.info(ex)
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            self.logger.info("predict value save!")

        self.logger.info('start')

        save_master()
        save_predict()

        self.logger.info('save')