import numpy as np
import time
from pathlib import Path
from pprint import pformat

import pandas as pd
import pathos
import psutil
import traceback

from algorithms.time_series_clustering import TimeSeriesClustering
from algorithms.dynamic_baseline_for_service import DynamicBaseline
from analyzer import aimodule
from api.tsa.tsa_utils import TSAUtils
from common import aicommon, constants
from common.error_code import Errors

# to prevent using only one core when multiprocessing
current_process = psutil.Process()
current_process.cpu_affinity(list(range(psutil.cpu_count(logical=True))))

# multiprocess batch size
default_multi_process_cnt = 4

FEATURES = {
    "2-tier": {
        "target_id_field": "xid",
        "target_name_field": "txn_name",
        "anomaly_target_id_field": "txns",
        "anomaly_target_name_field": "txn_names",
    },
    "e2e": {
        "target_id_field": "tx_code",
        "target_name_field": "tx_code_name",
        "anomaly_target_id_field": "tx_codes",
        "anomaly_target_name_field": "tx_code_names",
    },
    "service": {
        "target_id_field": "target_id",
        "target_name_field": "tx_code_name",
        "anomaly_target_id_field": "tx_codes",
        "anomaly_target_name_field": "tx_code_names",
    },
}


class ExemAiopsAnlsServiceService(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        E2E 시스템에 대한 이상탐지/분석 기능

        - 각 서비스 별 이상 탐지를 dynamic baseline으로 탐지하여 결과를 알려줌
        - 서비스 리스트를 서버로 부터 정보를 가져옴
        - 모델은 각 서비스 마다 저장함.

        Parameters
        ----------
        config
        logger
        """
        self.config = config
        self.logger = logger

        # naming rule에 따라 module/model에서 사용할 ID 생성
        # 이 ID를 이용해 csv 파일이나 model binary를 읽고 쓸때 필요한 이름 생성
        self.logger.info(f"config :{pformat(config)}")

        self.module_id = config["module"]
        self.sys_id = config["sys_id"]
        self.inst_type = config["inst_type"]
        self.target_id = config["target_id"]
        self.service_id = f"{self.inst_type}_{self.target_id}"

        # 분석 대상 서비스 리스트, 서버에서 설정하여 전달.
        self.service_list = config.get("service_list", [])

        self.target_id_field = FEATURES[self.inst_type]["target_id_field"]
        self.target_name_field = FEATURES[self.inst_type]["target_name_field"]
        self.anomaly_target_id_field = FEATURES[self.inst_type][
            "anomaly_target_id_field"
        ]
        self.anomaly_target_name_field = FEATURES[self.inst_type][
            "anomaly_target_name_field"
        ]

        # 병렬처리 정보
        self.is_multiprocessing_mode = True
        self.number_of_child_processes = int(psutil.cpu_count(logical=True) * 0.3)
        if self.number_of_child_processes <= 0:
            self.number_of_child_processes = 1
        self.multi_train_txn_threshold = 5  # 병렬 처리 학습이 가능한 최소 txn 개수

        # 분석 모델 인스턴스 생성
        self.dbsln = DynamicBaseline(self.service_id, self.config, self.logger)
        self.tsa = TSAUtils()
        self.multiprocessing_serving_flag = False

        self.tsClustering = TimeSeriesClustering(config, logger)

        self.init_param(config)
        self.empty_data_target_dict = {}

    # 서비스 설정 파라미터 변경
    def init_param(self, config):
        # dbsln config 초기화
        self.dbsln.init_param(config)

        return True

    def _save(self, training_model=None, train_result=None):
        model_dir = self.config["model_dir"]
        Path(model_dir).mkdir(exist_ok=True, parents=True)

        if self.is_multiprocessing_mode:
            self.dbsln.save_multi_mode(
                model_dir, training_model, train_result
            )
        else:
            self.dbsln.save(model_dir)

    def select_train_features(self, train_df):
        no_train_feats = list()
        for feat in train_df.columns:
            if feat not in self.config["parameter"]["train"]["dbsln"]["features"]:
                no_train_feats.append(feat)
        self.logger.info(f"[select_train_features] no_train_feats : {no_train_feats}")
        train_df = train_df.drop(no_train_feats, axis=1)
        train_df = train_df[train_df.index.notnull()]

        return train_df

    def train(self, train_logger):
        train_start_time = time.time()
        self.logger.info(f"module {self.service_id} start training")

        train_prog = {constants.PROCESS_TRAIN_DATA: self.default_module_status(),
                      constants.MODEL_F_DBSLN_FOR_SERVICE: self.default_module_status()}

        df = None
        try:
            df = self.tsa.data_loading(self.config, self.logger)
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[TSA] data_loading Exception: {tb}")

        aicommon.Query.update_module_status_by_training_id(
            self.config['db_conn_str'],
            self.config['train_history_id'],
            train_prog,
            train_process=constants.PROCESS_TRAIN_DATA,
            progress=100,
            process_start_time=f"{time.time() - train_start_time:.3f}"
        )

        total_train_result = {}
        Not_None_target_list = list()
        train_results = {}
        xcode_df_mapper = {}
        empty_data_target_list = list()
        biz_xcode_df_mapper = {}
        cnt = 0
        clustering_res_state = False
        except_clustering_date_list = dict()

        biz_df_1day_all = None
        try:
            biz_df_1day_all = self.tsa.business_data_loading(self.config)
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[TSA] business_data_loading Exception: {tb}")

        for xcode in self.service_list:
            self.logger.info(f"[training] preprocessing data of xcode no.{xcode}")

            if len(df) > 0:
                df_xcd = df.query(f"{self.target_id_field} == '{xcode}'")
                if len(df_xcd) == 0:
                    empty_data_target_list.append(xcode)

                if self.config.get('clustering_mode', False):
                    df_xcd, exclude_date = self.tsClustering.training(df_xcd, 'dbsln', multivariate=False)
                    if len(exclude_date) != 0:
                        clustering_res_state = True
                        except_clustering_date_list[xcode] = exclude_date
                    else:
                        except_clustering_date_list[xcode] = []

                # df_xcd.drop([self.target_id_field, 'total_elapse_time'], axis=1, inplace=True)
                df_xcd = df_xcd.set_index("time")
                df_xcd = self.select_train_features(df_xcd)
                df_xcd = df_xcd.interpolate()
                # df_xcd[df_xcd < 0] = np.nan
                xcode_df_mapper[xcode] = df_xcd
                biz_df_xcd_dict = None
                if biz_df_1day_all is not None:
                    biz_df_xcd_dict = dict()
                    for idx in biz_df_1day_all.keys():
                        biz_idx_df_1day_all = biz_df_1day_all[idx]
                        biz_df_xcd = biz_idx_df_1day_all.query(f"{self.target_id_field} == '{xcode}'")
                        if len(biz_df_xcd) != 0:
                            biz_df_xcd = biz_df_xcd.set_index("time")
                            biz_df_xcd = self.select_train_features(biz_df_xcd)
                            biz_df_xcd = biz_df_xcd.interpolate()
                            biz_df_xcd_dict[idx] = biz_df_xcd
                    biz_xcode_df_mapper[xcode] = biz_df_xcd_dict
                # 싱글 프로세스 학습 처리
                if self.is_multiprocessing_mode is False:
                    dbsln_result, _, error_code, error_message = self.dbsln.fit(
                        xcode, df_xcd, biz_df_xcd_dict, multiprocessing=False
                    )
                    if dbsln_result is not None:
                        train_results[xcode] = dbsln_result
                        Not_None_target_list.append(xcode)
                    cnt = cnt + 1

                    aicommon.Query.update_module_status_by_training_id(
                        self.config['db_conn_str'],
                        self.config['train_history_id'],
                        train_prog,
                        train_process=constants.MODEL_F_DBSLN_FOR_SERVICE,
                        progress=int((cnt / len(self.service_list) * 100)),
                    )

                    if len(Not_None_target_list) != 0:
                        total_train_result["dbsln"] = train_results[Not_None_target_list[0]]
                    else:
                        total_train_result["dbsln"] = None
        self.empty_data_target_dict["empty_data_target_list"] = empty_data_target_list

        # 멀티 프로세스 학습 처리
        if self.is_multiprocessing_mode:
            pool = pathos.multiprocessing.Pool(processes=self.number_of_child_processes)
            self.logger.info(
                f"[training] pool created with {self.number_of_child_processes} child processes"
            )

            # 병렬 학습 준비
            input_iterable = [(key, value, biz_xcode_df_mapper[key] if key in biz_xcode_df_mapper.keys() else None) for
                              key, value in xcode_df_mapper.items()]
            chunk_size, remainder = divmod(
                len(input_iterable), self.number_of_child_processes
            )
            if remainder != 0:
                chunk_size += 1

            self.logger.info(
                "[training] start multiprocessing. check train process log"
            )

            # 병렬 학습
            multi_proc_train_results = {}
            txn_models = list()
            multi_train_results = list()
            error_code = list()
            error_message = list()
            target_list = list()
            bizday_train_result = dict()
            for result in pool.starmap(self.dbsln.fit, input_iterable, chunksize=chunk_size):
                txn_models.append(result[0])
                multi_train_results.append(result[1])
                error_code.append(result[3])
                error_message.append(result[4])
                target_list.append(result[5])
                bizday_train_result[result[5]] = result[6]

                aicommon.Query.update_module_status_by_training_id(
                    self.config['db_conn_str'],
                    self.config['train_history_id'],
                    train_prog,
                    train_process=constants.MODEL_F_DBSLN_FOR_SERVICE,
                    progress=int((cnt / len(self.service_list) * 100)),
                )

            for i in range(len(target_list)):
                if multi_train_results[i] is not None:
                    multi_proc_train_results[target_list[i]] = multi_train_results[i]
                    Not_None_target_list.append(target_list[i])
            if len(Not_None_target_list) != 0:
                total_train_result["dbsln"] = multi_proc_train_results[Not_None_target_list[0]]
            else:
                total_train_result["dbsln"] = None

            pool.close()
            pool.join()

            self.logger.info("[training] finish multiprocessing")

            if error_code.count(0) < len(error_code):
                error_info = pd.DataFrame(
                    zip(target_list, error_code, error_message),
                    columns=["xcode", "error_code", "error_message"],
                )
                error_info.dropna(inplace=True)
                self.logger.error("description of error occured while multiprocessing")
                self.logger.error(f"{error_info.to_dict(orient='record')}")

            models = {}
            for element in txn_models:
                if element is None:
                    continue
                else:
                    for xcode in list(element.keys()):
                        models[xcode] = element[xcode]
            # 멀티 프로세스 학습 저장
            self._save(
                training_model=models,
                train_result=multi_proc_train_results,
            )

            duration_time = round(time.time() - train_start_time)
            untrained_target_list = list(set(self.service_list) - set(models.keys()))

            total_train_result['dbsln']['train_business_status'] = bizday_train_result  # train_business_status
            total_train_result['dbsln']['clustering_mode'] = clustering_res_state
            total_train_result['dbsln']['except_clustering_date_list'] = except_clustering_date_list
            total_train_result["dbsln"]["results"]["duration_time"] = duration_time
            total_train_result["dbsln"]["train_target_list"] = np.unique(list(models.keys()))
            total_train_result["dbsln"]["untrained_target_list"] = untrained_target_list
            self.logger.info(f"[training] training result: {total_train_result}")

            aicommon.Query.update_module_status_by_training_id(
                self.config['db_conn_str'],
                self.config['train_history_id'],
                train_prog,
                train_process=constants.MODEL_F_DBSLN_FOR_SERVICE,
                progress=100,
                process_start_time=duration_time
            )

            return total_train_result, None, 0, None

        # 싱글 프로세스 학습 저장
        self._save()
        duration_time = round(time.time() - train_start_time)

        total_train_result['dbsln']['clustering_mode'] = clustering_res_state
        total_train_result['dbsln']['except_clustering_date_list'] = except_clustering_date_list
        total_train_result["dbsln"]["results"]["duration_time"] = duration_time
        self.logger.info(f"[training] training result: {total_train_result}")

        aicommon.Query.update_module_status_by_training_id(
            self.config['db_conn_str'],
            self.config['train_history_id'],
            train_prog,
            train_process=constants.MODEL_F_DBSLN_FOR_SERVICE,
            progress=100,
            process_start_time=duration_time
        )

        return total_train_result, None, 0, None

    def test_train(self):
        pass

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload=False):
        pass

    def serve(self, header, data_dict):
        pass

    def end_serve(self):
        pass

    # API for dev and debug
    def get_debug_info(self):
        pass
