import time
import traceback
from pathlib import Path
from pprint import pformat

from algorithms.tsmixer.tsmixer import TSMixer
from analyzer import aimodule
from api.tsa.tsa_utils import TSAUtils
from common import aicommon, constants
from common.error_code import Errors

class ExemAiopsFcstTsmixer(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        instance의 "부하 예측" 를 수행
        기능
         - 학습
            - 입력 데이터 : 사용자에게 입력 받은 csv 형태의 pandas 의 지표별 시계열 데이터
            - 출력 데이터 : 데이터를 학습한 모델 파일과 학습 결과 파일
                - model_config.json, model_meta.json
         - 서빙
            - 입력 데이터 : 분당 t - 60 ~ t 까지의 각 인스턴스의 지표 데이터를 받아옴
            - 출력 데이터 : 이상 탐지 및 예측 결과를 dict 형태로 전달
                - 서버에서 db(postresql)로 전송함

        부하 예측 알고리즘
        - TSMixer
            - 타겟(다변량) 부하 예측

        설정
        - 서버에서 학습 및 서빙 여부를 json 파라미터로 받아옴.
             - serverAPI
        메뉴
         - 학습 : 설정 -> 학습/서비스 -> 학습 -> 타입 (was, db, host(os), ... etc) -> 모듈 (부하예측)
         - 차트 : 대시보드 -> 부하 예측 모니터링 or 대시보드 -> 부하 예측 모니터링

        Parameters
        ----------
        config : 각종 설정 파일, 추가로 서버에서 설정 파일을 받아 옴.
        logger : 로그를 출력하기 위한 로거
        """
        self.config = config
        self.logger = logger

        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.service_id = f"{self.module_id}_{config['target_id']}"

        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]

        self.logger.info(f"config :{pformat(config)}")

        self.tsmixer = None
        self.tsa = TSAUtils()

        self.model_dict = {}

        try:
            self.tsmixer = TSMixer(self.service_id, config, logger)

        except MemoryError as error:
            logger.exception(f"[Error] Unexpected memory error during serving : {error}")
            aicommon.Utils.print_memory_usage(logger)
            raise error
        except Exception as exception:
            logger.exception(f"[Error] Unexpected exception during serving : {exception}")
            raise exception


    def select_train_features(self, train_df):
        no_train_feats = list()
        for feat in train_df.columns:
            if feat not in self.config["parameter"]["train"]["tsmixer"]["features"]:
                no_train_feats.append(feat)
        self.logger.info(f"[select_train_features] no_train_feats : {no_train_feats}")
        train_df = train_df.drop(no_train_feats, axis=1)
        train_df = train_df[train_df.index.notnull()]

        return train_df

    def train(self, train_logger):
        self.logger.info(f"module {self.service_id} start training")

        train_prog = {
            'process train data': self.default_module_status(),
        }

        tsmixer_progress = {
            self.tsmixer.progress_name: self.default_module_status()
        }
        train_prog = {**train_prog, **tsmixer_progress}
        process_start_time = time.time()

        df = None
        try:
            df = self.tsa.data_loading(self.config, self.logger)
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[TSA] data_loading Exception: {tb}")

        df = df.set_index("time")
        meta_features = self.config["parameter"]["train"]["tsmixer"]["features"]
        if set(meta_features).issubset(df.columns):
            df = df[meta_features]
            df.interpolate(limit_area="inside", inplace=True)
            df.interpolate(limit_direction="both", inplace=True)
        else:
            missing_features = set(meta_features) - set(df.columns)
            self.logger.info(f"[training] no train data feature :{missing_features}")
            return None, None, Errors.E804.value, f" '{missing_features}' " + Errors.E804.desc
        
        if df is None or len(df) == 0:
            self.logger.info("[training] no train data")
            return None, None, Errors.E800.value, Errors.E800.desc

        self.logger.info(f"df : {df}")
        self.logger.info(f"df cols : {df.columns}")

        meta_features = self.config["parameter"]["train"]["tsmixer"]["features"]
        for col in df[meta_features]:
            if df[col].isnull().sum() == len(df):
                self.logger.info(f"[training] no train data feature :{col}")
                return None, None, Errors.E804.value, f" '{col}' " + Errors.E804.desc

        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'],
                                                           train_prog, constants.PROCESS_TRAIN_DATA, 100, f"{time.time() - process_start_time:.3f}")

        self.logger.debug(f"training (raw)data({df.shape}):\n{df}")

        tsmixer_result, body, error_code, error_msg = self.tsmixer.fit(df, train_progress=train_prog)

        self.logger.info(f"module {self.service_id} training result: {tsmixer_result}")

        return tsmixer_result, None, 0, None
