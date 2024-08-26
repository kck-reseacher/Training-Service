import datetime
import json
import os
import time
import traceback
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from algorithms.aimodel import TrainProgress
from algorithms.gdn.graph_detection_network import GraphDetectionNetwork
from analyzer import aimodule
from api.tsa.tsa_utils import TSAUtils
from common import aicommon, constants
from common.error_code import Errors
from common.module_exception import ModuleException


# window serving data 전역 변수
past_input_dict = dict()
_past_input_dict = dict()


class ExemAiopsAnlsInst(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        instance의 각 지표(metric)의 "이상 탐지"와 "예측"을 함
        기능
         - 학습
            - 입력 데이터 : 사용자에게 입력 받은 csv 형태의 pandas 의 지표별 시계열 데이터
            - 출력 데이터 : 데이터를 학습한 모델 파일과 학습 결과 파일
                - model_config.json, model_meta.json
         - 서빙
            - 입력 데이터 : 분당 t - 60 ~ t 까지의 각 인스턴스의 지표 데이터를 받아옴
            - 출력 데이터 : 이상 탐지 및 예측 결과를 dict 형태로 전달
                - 서버에서 db(postresql)로 전송함
        지원 알고리즘
         - 이상 탐지
             - GDN
             - SeqAttn 04-18부 제거

        메뉴
         - 학습 : 설정 -> 학습/서비스 -> 학습 -> 타입 (was, db, host(os), ... etc) -> 모듈 (이상탐지 / 부하예측)
         - 서비스 : 설정 -> 학습/서비스 -> 서비스 -> 타입 (was, db, host(os), ... etc) -> 모듈 (이상탐지 / 부하예측)
         - 차트 : 대시보드 -> 이상 탐지 모니터링 or 대시보드 -> 부하 예측 모니터링

        Parameters
        ----------
        config : 각종 설정 파일, 추가로 서버에서 설정 파일을 받아 옴.
        logger : 로그를 출력하기 위한 로거
        """
        self.config = config
        self.logger = logger

        # naming rule에 따라 module/model에서 사용할 ID 생성
        # 이 ID를 이용해 csv 파일이나 model binary를 읽고 쓸때 필요한 이름 생성
        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.service_id = f"{self.module_id}_{config['target_id']}"

        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]

        self.logger.info(f"config :{pformat(config)}")

        # set parameters
        # TODO 신규 알고리즘 테스트
        self.use_gdn = constants.USE_GDN and config["parameter"]["train"].get("gdn", {}).get("use", False)
        self.gdn = None

        self.tsa = TSAUtils()
        try:
            if self.use_gdn:
                self.gdn = GraphDetectionNetwork(self.service_id, config, logger)

        except MemoryError as error:
            logger.exception(
                f"[Error] Unexpected memory error during serving : {error}"
            )
            aicommon.Utils.print_memory_usage(logger)
            raise error
        except Exception as exception:
            logger.exception(
                f"[Error] Unexpected exception during serving : {exception}"
            )
            raise exception

    def select_train_features(self, train_df):
        no_train_feats = list()
        for feat in train_df.columns:
            if feat not in self.config["parameter"]["train"]["gdn"]["features"]:
                no_train_feats.append(feat)
        self.logger.info(f"[select_train_features] no_train_feats : {no_train_feats}")
        train_df = train_df.drop(no_train_feats, axis=1)
        train_df = train_df[train_df.index.notnull()]

        return train_df

    # API for training
    def train(self, train_logger):
        self.logger.info(f"module {self.service_id} start training")

        train_prog = {'process train data': self.default_module_status()}

        if self.use_gdn:
            gdn_progress = {
                self.gdn.progress_name: self.default_module_status()
            }
            train_prog = {**train_prog, **gdn_progress}

        process_start_time = time.time()

        df = None
        try:
            df = self.tsa.data_loading(self.config, self.logger)
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[TSA] data loading Exception: {tb}")

        if df is None or len(df) == 0:
            self.logger.info("[training] no train data")
            return None, None, Errors.E800.value, Errors.E800.desc

        for col in df.columns:
            if df[col].isnull().sum() == len(df):
                self.logger.info(f"[training] no train data feature :{col}")
                return None, None, Errors.E804.value, f" '{col}' " + Errors.E804.desc

        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'],
                                                           train_prog, constants.PROCESS_TRAIN_DATA, 100, f"{time.time() - process_start_time:.3f}")
        self.logger.debug(f"training (raw)data({df.shape}):\n{df}")

        for algo, info in self.config["parameter"]["train"].items():
            self.config["parameter"]["train"][algo]["features"] = list(set(info["features"]) & set(df.columns))

        # make train dataframe
        temp_df = df.copy()
        temp_df = temp_df.set_index("time")
        temp_df = self.select_train_features(temp_df)
        temp_df[temp_df < 0] = np.nan
        temp_df = temp_df.interpolate().bfill().ffill()

        rows, columns = temp_df.shape
        if rows == 0:
            self.logger.error("[training] no train data")
            return None, None, Errors.E800.value, Errors.E800.desc
        if columns == 0:
            self.logger.error("[training] training features are not provided")
            return None, None, Errors.E803.value, Errors.E803.desc

        # gdn 이상탐지 학습
        if self.use_gdn:
            try:
                gdn_result = self.gdn.fit(temp_df, train_progress=train_prog)
                self.logger.info(f"Success [gdn] training ")

            except Exception:
                tb = traceback.format_exc()
                self.logger.info(f"Fail [gdn] training :{tb}")
                return None, None, Errors.E833.value, Errors.E833.desc

        # train_result
        result = {}
        if self.use_gdn:
            result["gdn"] = gdn_result

        self.logger.info(f"module {self.service_id} training result: {result}")

        return result, None, 0, None

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload=False):
        pass

    def _predict_by_algo(self, model, algo, **kwargs):
        pass

    def serve(self):
        pass

    def end_serve(self):
        pass

    # API for dev and debug
    def get_debug_info(self):
        pass
