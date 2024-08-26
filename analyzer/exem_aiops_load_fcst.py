import datetime
import json
import os
import time
import traceback
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

from algorithms.load_forecast import LoadForecast
from analyzer import aimodule
from api.tsa.tsa_utils import TSAUtils
from common import aicommon, constants
from common.error_code import Errors
from common.module_exception import ModuleException


class ExemAiopsLoadFcst(aimodule.AIModule):
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
         - 시계열 예측
             - Seq2Seq
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
        self.service_id = f"{self.module_id}_{config['target_id']}"

        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]

        self.logger.info(f"config :{pformat(config)}")

        self.load_forecast = None

        self.tsa = TSAUtils()

        self.model_dict = {}

        try:
            self.load_forecast = LoadForecast(self.service_id, config, logger)
            self.load_forecast.init_config(config)
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

        self.load_forecast.save(model_dir)

    def _load(self, reload=False):
        pass

    def select_train_features(self, train_df):
        no_train_feats = list()
        for feat in train_df.columns:
            if feat not in self.config["parameter"]["train"]["seq2seq"]["features"]:
                no_train_feats.append(feat)
        self.logger.info(f"[select_train_features] no_train_feats : {no_train_feats}")
        train_df = train_df.drop(no_train_feats, axis=1)
        train_df = train_df[train_df.index.notnull()]

        return train_df

    def get_dataframe(self, base_dir, target_list):
        result = []
        for i in target_list:
            train_dir = base_dir + "/" + i
            csv_list = self.get_csvfile_list_all(
                train_dir,
                self.config["date"][0],
                self.config["date"][-1],
            )
            df = self.load_csvfiles(csv_list)
            # self.logger.info(f"target: {i}, df type : {type(df)}")
            if isinstance(df, pd.DataFrame):
                result.append(df)

        self.logger.info(f"result : {result}")
        if len(result) == 0:
            return result
        return pd.concat(result, ignore_index=True)

    # API for training
    def train(self, train_logger):
        self.logger.info(f"module {self.service_id} start training")

        train_prog = {'process train data': self.default_module_status(),
                      self.load_forecast.progress_name: self.default_module_status()}

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
        elif df.shape[0] < constants.DBSLN_DMIN_MAX * 2:
            return None, None, Errors.E801.value, Errors.E801.desc

        self.logger.info(f"df : {df}")
        self.logger.info(f"df cols : {df.columns}")

        for col in df.columns:
            if df[col].isnull().sum() == len(df):
                self.logger.info("[training] no train data feature")
                del df[col]
                # return None, None, Errors.E804.value, f" '{col}' " + Errors.E804.desc

        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'],
                                                           train_prog, constants.PROCESS_TRAIN_DATA, 100,
                                                           f"{time.time() - process_start_time:.3f}")
        self.logger.debug(f"training (raw)data({df.shape}):\n{df}")
        self.logger.debug(f"training (processed)data({df.shape}):\n{df}")

        if len(self.config["parameter"]["train"]["seq2seq"]["features"]) == 0:
            self.logger.error("[training] training features are not provided")
            return None, None, Errors.E803.value, Errors.E803.desc

        train_features = []
        for feat in self.config["parameter"]["train"]["seq2seq"]["features"]:
            if feat in df.columns:
                train_features.append(feat)
        self.logger.info(f"train_features : {train_features}")

        df = df.loc[:, train_features + ["time", "target_id"]].copy()
        df = df.set_index("time")

        df["target_id"] = df["target_id"].astype(str)

        # load businessday training data
        self.load_forecast.init_config(self.config)
        # update pred_feats
        self.load_forecast.pred_feats = train_features
        load_forecast_result, body, error_code, error_msg = self.load_forecast.fit(df, train_progress=train_prog)

        # train_result
        result = {"seq2seq": load_forecast_result}

        self.logger.info(f"module {self.service_id} training result: {result}")

        return result, None, 0, None

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload=False):
        pass

    def serve(self, header, data_dict):
        pass

    def end_serve(self):
        pass