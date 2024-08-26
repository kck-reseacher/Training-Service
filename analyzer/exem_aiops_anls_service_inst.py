import copy
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
from algorithms.dynamic_baseline2 import DynamicBaseline
from algorithms.time_series_clustering import TimeSeriesClustering

from analyzer import aimodule
from api.tsa.tsa_utils import TSAUtils
from common import aicommon, constants
from common.error_code import Errors
from common.module_exception import ModuleException


# window serving data 전역 변수
past_input_dict = dict()
_past_input_dict = dict()


class ExemAiopsAnlsServiceInst(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        instance의 "이상 탐지" 를 수행
        기능
         - 학습
            - 입력 데이터 : 사용자에게 입력 받은 csv 형태의 pandas 의 지표별 시계열 데이터
            - 출력 데이터 : 데이터를 학습한 모델 파일과 학습 결과 파일
                - model_config.json, model_meta.json
         - 서빙
            - 입력 데이터 : 분당 t - 60 ~ t 까지의 각 인스턴스의 지표 데이터를 받아옴
            - 출력 데이터 : 이상 탐지 및 예측 결과를 dict 형태로 전달
                - 서버에서 db(postresql)로 전송함
        이상 탐지 알고리즘
         - Dynamic Baseline
            - 각 지표 metric 개별 이상탐지

        설정
        - 서버에서 학습 및 서빙 여부를 json 파라미터로 받아옴.
             - serverAPI
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
        self.service_id = f"{self.inst_type}_all_{config['target_id']}"

        self.sys_id = config["sys_id"]

        self.logger.info(f"config :{pformat(config)}")

        # set parameters
        self.dbsln = None

        self.tsClustering = TimeSeriesClustering(config, logger)
        self.tsa = TSAUtils()

        self.model_dict = {}

        try:
            self.dbsln = DynamicBaseline(self.service_id, config, logger)
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

    def _save(self):
        model_dir = self.config["model_dir"]
        Path(model_dir).mkdir(exist_ok=True, parents=True)

        self.dbsln.save(model_dir)

    # 사용자에 의해 config 변경시 새로운 서버 알림으로 config로 초기화
    def init_param(self, config):
        self.dbsln.init_param(config)
        return True

    def select_train_features(self, train_df):
        no_train_feats = list()
        for feat in train_df.columns:
            if feat not in self.config["parameter"]["train"]["dbsln"]["features"]:
                no_train_feats.append(feat)
        self.logger.info(f"[select_train_features] no_train_feats : {no_train_feats}")
        train_df = train_df.drop(no_train_feats, axis=1)
        train_df = train_df[train_df.index.notnull()]

        return train_df

    # API for training
    def train(self):
        self.logger.info(f"module {self.service_id} start training")

        process_start_time = time.time()

        df = None
        try:
            df = self.tsa.data_loading(self.config, self.logger)
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[TSA] data_loading Exception: {tb}")

        if df is None or len(df) == 0:
            self.logger.info("[training] no train data")
            return None, None, Errors.E800.value, Errors.E800.desc

        meta_features = self.config["parameter"]["train"]["dbsln"]["features"]
        if set(meta_features).issubset(df.columns):
            for col in df[meta_features]:
                if df[col].isnull().sum() == len(df):
                    self.logger.info(f"[training] no train data feature :{col}")
                    return None, None, Errors.E804.value, f" '{col}' " + Errors.E804.desc
        else:
            missing_features = set(meta_features) - set(df.columns)
            self.logger.info(f"[training] no train data feature :{missing_features}")
            return None, None, Errors.E804.value, f" '{missing_features}' " + Errors.E804.desc

        # self.logger.debug(f"training (raw)data({df.shape}):\n{df}")

        for algo, info in self.config["parameter"]["train"].items():
            self.config["parameter"]["train"][algo]["features"] = list(set(info["features"]) & set(df.columns))
        self.init_param(self.config)

        clustering_res_state = False
        if self.config.get('clustering_mode', False):
            df, exclude_date = self.tsClustering.training(df, 'dbsln', multivariate=False)
            if len(exclude_date) != 0:
                clustering_res_state = True

        df_for_ad = df.copy()
        df_for_ad = df_for_ad.set_index("time")
        df_for_ad = self.select_train_features(df_for_ad)

        biz_df_1day_all = None
        try:
            biz_df_1day_all = self.tsa.business_data_loading(self.config)
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[TSA] business_data_loading Exception: {tb}")

        # dbsln 이상탐지 학습
        try:
            dbsln_result, body, code, message = self.dbsln.fit(df_for_ad, biz_df_1day_all)
            dbsln_result['clustering_mode'] = clustering_res_state
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.info(f"dbsln fit Exception: {tb}")
            exception_msg = f"{type(e).__name__}: {e}"
            return None, None, Errors.E910.value, f" '{exception_msg}' " + Errors.E910.desc

        if clustering_res_state:
            dbsln_result['except_clustering_date_list'] = exclude_date

        self._save()

        # train_result
        result = {"dbsln": dbsln_result}

        self.logger.info(f"module {self.service_id} training result: {result}")

        return result, None, 0, None

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload=False):
        pass

    def serve(self):
        pass

    def end_serve(self):
        pass

    # API for dev and debug
    def get_debug_info(self):
        pass
