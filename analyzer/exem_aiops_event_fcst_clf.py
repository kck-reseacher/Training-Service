import json
import time
import traceback
from collections import defaultdict
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from algorithms.mtad_gat.args import hyper_parameter
from algorithms.mtad_gat.mtad_gat import MTAD_GAT
from algorithms.mtad_gat.prediction import Predictor
from algorithms.mtad_gat.training import Trainer
from algorithms.mtad_gat.utils import *
from analyzer import aimodule
from common.constants import SystemConstants as sc
from common.system_util import SystemUtil
from common import aicommon, constants
from common import constants as bc
from common.aicommon import Query
from common.aicommon import sMAPE
from common.error_code import Errors
from common.onnx_torch import ONNXTorch
from common.timelogger import TimeLogger
from api.tsa.tsa_utils import TSAUtils
from algorithms.tsmixer.tsmixer import TSMixer
import pynvml
import tensorflow as tf
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
total_memory = info.total
# Get the number of MIG devices
num_mig_devices = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
os_env = SystemUtil.get_environment_variable()
if num_mig_devices > 0 and os_env[sc.GPU_MIG]:
    total_memory = total_memory // 2

memory_limit = (total_memory/(1024**2)) // 2
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

class ExemAiopsEventFcstClf(aimodule.AIModule):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.target_logger = None
        self.tsa = TSAUtils()
        self.db_query = Query.CreateQuery(self.config, self.logger)
        self.home = Path(config.get("home")) if config.get("home") else None
        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]
        self.inst_type = config["inst_type"]
        self.module_id = config["module"]
        self.log_dir = config["log_dir"]
        self.train_params = {}
        self.service_id = f"{self.module_id}_{self.inst_type}_{self.target_id}"
        # 서빙 시 이벤트 feature, msg 정보 담음
        self.event_cluster = None
        self.event_msg = None
        # codeGroup 서빙 시 이벤트 특정 위한 dict
        self.event_feature_dict = dict()

        # set model param
        hp = hyper_parameter
        # 학습에 이용할 데이터 조건
        self.least_train_days = hp["least_train_days"]
        self.least_nan_percent = hp["least_nan_percent"]
        # 에폭 배치 학습률은 대시보드 통한 입력으로 변경
        self.p_n_epochs = hp["p_n_epochs"]
        self.p_batch_size = hp["p_bs"]
        self.p_learning_rate = hp["p_init_lr"]
        # 이외의 param
        self.window_size = hp["lookback"]
        self.normalize = hp["normalize"]
        self.spec_res = hp["spec_res"]

        # Conv1D, GAT layers
        self.kernel_size = hp["kernel_size"]
        self.use_gatv2 = hp["use_gatv2"]
        self.feat_gat_embed_dim = hp["feat_gat_embed_dim"]
        self.time_gat_embed_dim = hp["time_gat_embed_dim"]
        # GRU
        self.gru_n_layers = hp["gru_n_layers"]
        self.gru_hid_dim = hp["gru_hid_dim"]

        self.fc_n_layers = hp["fc_n_layers"]
        self.fc_hid_dim = hp["fc_hid_dim"]

        self.recon_n_layers = hp["recon_n_layers"]
        self.recon_hid_dim = hp["recon_hid_dim"]

        self.alpha = hp["alpha"]

        self.val_split = hp["val_split"]
        self.shuffle_dataset = hp["shuffle_dataset"]
        self.dropout = hp["dropout"]
        self.use_cuda = hp["use_cuda"]
        self.device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        self.print_every = hp["print_every"]
        self.log_tensorboard = hp["log_tensorboard"]

        self.scale_scores = hp["scale_scores"]
        self.use_mov_av = hp["use_mov_av"]
        self.gamma = hp["gamma"]
        self.level = hp["level"]
        self.q = hp["q"]
        self.reg_level = hp["reg_level"]
        self.dynamic_pot = hp["dynamic_pot"]

        # Serving Param
        self.forecast_len = hp["forecast_len"]
        # Anomaly Forecast 시간 가중치
        self.weight_arr = None

        self.model_dir = self.config["model_dir"]

        self.models = {}
        self.bundle = {}
        self.model_name_list = list()

    def init_train(self):
        pass

    def train(self, train_logger, train_prog):
        """

        Args:
            train_logger:
            train_prog:

        Returns:

        """
        def remove_outlier_using_ma(data, mean_window_size=180, std_window_size=1440):
            """
            data : pd.DataFrame, 한 타겟의 전체 데이터
            """
            columns_name = data.columns
            mean_df = data.rolling(window=mean_window_size).mean().fillna(data)  # 이동 평균
            # std_df = data.rolling(window=std_window_size).std().fillna(0)  # 이동 편차, 전체 편차(data.std()) 사용 시 더 민감하게 이상치 제거됨
            std_df = data.std()

            overshoot = (mean_df + 3 * std_df)[data > (mean_df + 3 * std_df)]
            undershoot = (mean_df - 3 * std_df)[data < (mean_df - 3 * std_df)]
            merge_over_under = np.where(pd.notnull(overshoot) == True, overshoot, undershoot)
            clean_data = np.where(pd.notnull(merge_over_under) == True, merge_over_under, data)
            return pd.DataFrame(clean_data, columns=columns_name)
        self.logger.info(f"module {self.service_id} start training")
        desc = "EventPredictor"
        result = {
            desc: {
                "train_history_id": self.config["train_history_id"],
                "from_date": None,
                "to_date": None,
                "train_mode": -1,
                "results": {
                    "duration_time": 0,
                    "smape": 0,
                    "f1_score": 0,
                }
            }
        }
        #230627 add by jiwoo
        train_prog[desc] = aimodule.AIModule.default_module_status()
        train_prog[constants.PROCESS_TRAIN_DATA] = aimodule.AIModule.default_module_status()
        #train_prog = {
        #    desc: aimodule.AIModule.default_module_status(),
        #    constants.PROCESS_TRAIN_DATA: aimodule.AIModule.default_module_status()
        #}
        train_result = {}
        self.logger.debug(
            f"train.csv file is located {self.config['train_dir']}"
        )

        with TimeLogger(
                f"[{type(self).__name__}] fetching input data time :",
                self.logger,
        ):
            inst_type_list, target_id_list, db_type, event_list, event_cluster = self.parse_group_target()
            df_list, target_full_name_list, used_target_list, event_list, untrained_target_list = self.path_to_dataframe(inst_type_list, target_id_list, db_type, event_list)
        if target_full_name_list is None:
            return None, None, Errors.E800.value, Errors.E800.desc

        if len(event_list) == 0:
            return None, None, Errors.E801.value, Errors.E801.desc

        aicommon.Query.update_module_status_by_training_id(
            self.config['db_conn_str'],
            self.config['train_history_id'],
            train_prog,
            train_process=constants.PROCESS_TRAIN_DATA,
            progress=100)

        cum_duration = 0
        group_model_tuple_list = []  # [(was_214, model), (was_215, model), ]
        if len(target_full_name_list) == len(df_list) == len(event_list):
            self.logger.info(f"Number of targets name:{len(target_full_name_list)}, "
                             f"Number of targets dataframe: {len(df_list)}, "
                             f"Number of targets' event: {len(event_list)} ---- dataset is ready")
        else:
            msg = "dataset error"
            self.logger.exception(msg)
            raise Exception(msg)

        smape_list = []  # 타겟 별 smape 저장
        for cnt, (key, df, event) in enumerate(zip(target_full_name_list, df_list, event_list)):
            start = time.time()
            cols = df.columns

            tmp_config = {
                'train' : {'features' : cols},
                'module': 'exem_aiops_event_fcst',
                'sys_id': self.config['sys_id'],
                'inst_type': key.split('_')[0],
                'target_id': '_'.join(key.split('_')[1:]),
                'date': self.config['date'],
                'model_dir' : self.config["model_dir"],
                'train_history_id': self.config['train_history_id'],
                'except_failure_date_list': self.config['except_failure_date_list'],
                'outlier_mode': self.config['outlier_mode'],
                'clustering_mode': self.config['clustering_mode'],
                'regenerate_train_data': False
            }
            tmp_tsmixer= TSMixer(self.service_id, tmp_config, self.logger)

            train_data = self._fill_missing_value(df, mode=2)

            tmp_tsmixer_result, body, _, _ = tmp_tsmixer.fit(train_data, train_progress=train_prog, is_event_fcst=True)

            self.logger.info(f"============ {key}'s Predict Model Train Process Start ============")
            # 이상치 제거
            train_data = remove_outlier_using_ma(train_data, mean_window_size=180, std_window_size=1440)
            # MinMax 스케일
            x_train, scaler, columns = self._normalize_data(train_data)
            x_train = torch.from_numpy(x_train).float()
            n_features = x_train.shape[1]
            out_dim = n_features

            train_dataset = SlidingWindowDataset(x_train, self.window_size)

            train_loader, val_loader, test_loader = create_data_loaders(
                train_dataset, self.p_batch_size, self.val_split, self.shuffle_dataset
            )

            model = MTAD_GAT(
                n_features,
                self.window_size,
                out_dim,
                kernel_size=self.kernel_size,
                use_gatv2=self.use_gatv2,
                feat_gat_embed_dim=self.feat_gat_embed_dim,
                time_gat_embed_dim=self.time_gat_embed_dim,
                gru_n_layers=self.gru_n_layers,
                gru_hid_dim=self.gru_hid_dim,
                forecast_n_layers=self.fc_n_layers,
                forecast_hid_dim=self.fc_hid_dim,
                recon_n_layers=self.recon_n_layers,
                recon_hid_dim=self.recon_hid_dim,
                dropout=self.dropout,
                alpha=self.alpha,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=self.p_learning_rate)
            forecast_criterion = nn.MSELoss()
            recon_criterion = nn.MSELoss()
            p_trainer = Trainer(
                model,
                optimizer,
                self.window_size,
                n_features,
                None,
                self.p_n_epochs,
                self.p_batch_size,
                self.p_learning_rate,
                forecast_criterion,
                recon_criterion,
                self.use_cuda,
                key,
                self.logger,
                self.print_every,
                self.log_tensorboard,
            )
            p_trainer.fit(train_loader, val_loader)
            p_trainer.get_best_model()
            best_p_model = p_trainer.model

            self.logger.info(f"============ {key}'s Get Event Threshold & CDF Start ============")
            prediction_args = {
                "columns": df.columns.values.tolist(),
                "target_dims": None,
                'scale_scores': self.scale_scores,
                "level": self.level,
                "q": self.q,
                'dynamic_pot': self.dynamic_pot,
                "use_mov_av": self.use_mov_av,
                "gamma": self.gamma,
                "reg_level": self.reg_level,
                "use_cuda": self.use_cuda
            }

            predictor = Predictor(
                best_p_model,
                event,
                self.window_size,
                n_features,
                prediction_args,
            )

            threshold, anomaly_cdf, actual, recons = predictor.predict_anomalies(
                x_train)  # threshold는 mtad_gat/prediction.py의 threshold_50_percent
            group_model_tuple_list.append((key, best_p_model, scaler, columns, threshold, anomaly_cdf))

            # Make train result
            duration = time.time() - start

            inv_actual = scaler.inverse_transform(actual)
            inv_recons = scaler.inverse_transform(recons)
            smape = sMAPE(inv_actual, inv_recons)
            smape_list.append(smape)

            mse = ((inv_actual.reshape(-1) - inv_recons.reshape(-1)) ** 2).mean()

            train_result[key] = {
                "duration_time": duration,
                "smape": smape,
                "mse": mse,
                "f1_score": 0,
            }

            num = int(((cnt + 1) / len(target_full_name_list) * 100))
            cum_duration += duration
            aicommon.Query.update_module_status_by_training_id(
                self.config['db_conn_str'],
                self.config['train_history_id'],
                train_prog,
                train_process=desc,
                progress=num,
                process_start_time=f"{cum_duration:.3f}")

            self.logger.info(f"============ {key}'s ALL Process Done ============")

        total_smape = round(sum(smape_list) / len(smape_list), 2)
        result[desc]["results"]['duration_time'] = train_prog[desc][bc.DURATION_TIME]
        # total smape
        result[desc]["results"]["f1_score"] = 0
        result[desc]["results"]["smape"] = total_smape
        result[desc]["from_date"] = pd.to_datetime(self.config["date"][0], format=constants.INPUT_DATETIME_FORMAT)
        result[desc]["to_date"] = pd.to_datetime(self.config["date"][-1], format=constants.INPUT_DATETIME_FORMAT)
        result[desc]['train_metrics'] = train_result

        # target_metric
        target_metric = self.config["train"]["eventfcst"]["features"] if self.config.get("train") else None
        db_features_dict = dict()

        check = 0
        if 'db' in target_metric.keys():  # 인스턴스 그룹
            for k, v in target_metric["db"].items():  # ORA, TIBERO, PG
                if 'db' in used_target_list.keys():
                    if k in used_target_list["db"].keys():
                        if len(used_target_list["db"][k]) == 0:
                            continue
                        check += 1
                        db_features_dict[k] = v

            if check > 0:
                target_metric["db"] = db_features_dict
            else:
                del target_metric["db"]

        key_del = list()
        for k, v in target_metric.items():
            if k == "db":
                continue
            if k in used_target_list.keys():
                if len(used_target_list[k]) == 0:
                    key_del.append(k)
            else:
                key_del.append(k)

        for k in key_del:
            del target_metric[k]
        result['target_metric'] = target_metric

        # target_list
        db_target_list = list()
        if "db" in used_target_list.keys():
            for k, v in used_target_list["db"].items():
                db_target_list.extend(v)
            used_target_list["db"] = db_target_list
            if len(used_target_list["db"]) == 0:
                del used_target_list["db"]
        result['target_list'] = used_target_list

        # 이 외의 지표
        result['event_definition'] = self.config['event_definition'] if self.config.get('event_definition') else None
        result['event_cluster'] = event_cluster
        event_msg = dict()
        for k, v in self.config['event_definition'].items():
            event_msg[k] = v["msg"]
        result['event_msg'] = event_msg
        result['untrained_target_list'] = untrained_target_list

        self.logger.info(f"============ Model Save Process Start ============")
        self._save(group_model_tuple_list)

        self.logger.info(f"============ ALL Train Process Done ============")
        return result, None, 0, None

    def _read_train_meta(self, path):
        """
        read and return train_meta.json

        Args:
            path(string): path is the variables which hold train directory path
        Returns:
            train_meta(namedtuple): hold train_meta's info:
                train_meta.target_id,
                train_meta.inst_type,
                train_meta.from_date,
                train_meta.to_date,
        """
        try:
            file_json = json.loads((Path(path) / 'train_meta.json').read_text())
        except Exception as e:
            self.logger.error(f"can't read train_meta.json")
            return None

        train_meta = namedtuple(
            'train_meta',
            [
                'target_id',
                'inst_type',
                "date",
                'from_date',
                'to_date',
                "inst_type_list",
                "target_id_list",
                "group_target_list",
                "failure_label_list",
                "group_features",
                "event_definition"
            ]
        )

        return train_meta(
            file_json['target_id'] if file_json.get('target_id') else None,
            file_json['inst_type'] if file_json.get('inst_type') else None,
            [str(date) + '.csv' for date in file_json['date']] if file_json.get('date') else None,
            pd.to_datetime(file_json['date'][0], format=constants.INPUT_DATETIME_FORMAT),
            pd.to_datetime(file_json['date'][-1], format=constants.INPUT_DATETIME_FORMAT),
            file_json['inst_type_list'] if file_json.get('inst_type_list') else None,
            file_json['target_id_list'] if file_json.get('target_id_list') else None,
            file_json['group_target_list'] if file_json.get("group_target_list") else None,
            file_json['failure_label_list'] if file_json.get('failure_label_list') else None,
            file_json['train']['eventfcst']["features"] if file_json.get('train') else None,
            file_json['event_definition'] if file_json.get("event_definition") else None,
        )

    def parse_group_target(self):
        target_list = self.config['group_target_list'] if self.config.get('group_target_list') else None
        target_list = target_list if isinstance(target_list, dict) else json.loads(target_list)

        event_dict = self.config['event_definition'] if self.config.get('event_definition') else None
        event_cluster = defaultdict(dict)
        db_event_cluster = defaultdict(dict)

        for key, value in event_dict.items():
            for k, v in value["features"].items():
                if k == "db":
                    for kk, vv in v.items():
                        db_event_cluster[kk][key] = vv
                else:
                    event_cluster[k][key] = v
        event_cluster["db"] = db_event_cluster
        inst_type_list = []
        target_id_list = []
        event_list = []
        db_type = {}
        for inst_type in target_list.keys():
            if inst_type == "db":
                for temp_type in target_list[inst_type]:
                    for target_id in target_list[inst_type][temp_type]:
                        inst_type_list.append(inst_type)
                        target_id_list.append(target_id)
                        db_type.setdefault(target_id, temp_type)
                        event_list.append(event_cluster[inst_type][temp_type])
            else:
                for target_id in target_list[inst_type]:
                    inst_type_list.append(inst_type)
                    target_id_list.append(target_id)
                    event_list.append(event_cluster[inst_type])

        return inst_type_list, target_id_list, db_type, event_list, dict(event_cluster)

    def path_to_dataframe(self, inst_type_list, target_id_list, db_type, event_list):
        self.logger.debug(f"path_to_dataframe method start")
        used_target_list = deepcopy(self.config['group_target_list'] if self.config.get('group_target_list') else None)
        untrained_target_list = []  # 데이터 부족 학습 안하는 타겟 정보 제공
        train_amount_dict = dict()  # 타겟이 학습을 진행한 일자
        df_list = list()
        target_full_name_list = list()
        n_event = len(event_list)  # 데이터 없는 경우 event_list 삭제할 index 위한 변수
        for index, (inst_type, target_id) in enumerate(zip(inst_type_list, target_id_list)):
            untrained_target_list.append(target_id)
            df_stack = pd.DataFrame()
            df = None
            try:
                if inst_type == 'db':
                    df = self.tsa.data_loading(self.config, self.logger, db_type, target_id)
                else:
                    df = self.tsa.data_loading(self.config, self.logger, inst_type, target_id)
            except Exception:
                tb = traceback.format_exc()
                self.logger.info(f"[TSA] data_loading exception: {tb}")
            df.loc[:, 'datetime'] = pd.to_datetime(df['time'])
            training_days = 0
            for date, date_df in df.groupby(df['datetime'].dt.date):
                date_df = date_df.drop(columns=['datetime'])
                date_df = date_df.set_index('time')
                date_df = date_df.loc[~date_df.index.duplicated(keep='first')]
                # least_nan_percent 이상 결측날은 학습에서 제외
                train_df, df_frame = self.check_missing_data(date_df)

                if (train_df.isna().sum() > len(df_frame) * self.least_nan_percent).sum() > 0:
                    self.logger.warning(f"{train_df.isna().sum()} of data is missing in [date: {date}, target_id: {target_id}]")
                    continue
                training_days += 1
                df_stack = pd.concat([df_stack, train_df])
            # df 혹은 df의 특정 컬럼이 모두 NULL인 경우
            if len(df_stack) == 0 or (
                    df_stack.isnull().sum() == len(df_stack)).sum() > 0 or training_days < self.least_train_days:
                self.logger.warning(f"{inst_type}_{target_id} validation check failed.")
                if str(inst_type) != "db":
                    used_target_list[inst_type].remove(target_id)
                else:
                    used_target_list[inst_type][db_type[str(target_id)]].remove(target_id)
                # 데이터 없는 것을 반영해 event_list 수정(요소 제거)
                del event_list[index - n_event]
                train_amount_dict[target_id] = training_days
                continue

            # 학습 진행 타겟은 리스트 안에서 제거
            untrained_target_list.remove(target_id)
            train_amount_dict[target_id] = training_days
            df_list.append(df_stack)
            target_full_name_list.append(f"{inst_type}_{target_id}")
        self.logger.info(f"number of training days of each target: {train_amount_dict}")  # 타겟이 학습한 날짜 정보
        return df_list, target_full_name_list, used_target_list, event_list, untrained_target_list

    def check_missing_data(self, df):
        df["daytime"] = pd.to_datetime(df.index)
        df.index = df["daytime"]
        n_date = str(df["daytime"][0].date())
        t_index = pd.DatetimeIndex(pd.date_range(start=n_date, end=f'{n_date} 23:59:00', freq="1min"))
        df_frame = df.resample('1min', on='daytime').count().reindex(t_index).fillna(0)
        if "daytime" in df_frame.columns:
            df_frame = df_frame.drop(columns="daytime", axis=1)
        df_frame = df_frame.replace(0, None)
        df = df.drop(columns="daytime", axis=1)
        train_df = pd.concat([df_frame, df], axis=1).iloc[:, len(df.columns):]
        return train_df, df_frame

    def read_file(self, file_name, features):
        try:
            return pd.read_csv(file_name, usecols=features+["time"])
        except Exception as e:
            self.logger.warning(f"fail to load file: [{file_name}] caused by {e}")
            return pd.DataFrame()

    def _fill_missing_value(self, df, mode=1):
        self.logger.debug(f"_fill_missing_value method start")
        return df.interpolate(method="linear").fillna(method='ffill').fillna(method='bfill').fillna(0)

    def _normalize_data(self, data):
        self.logger.debug(f"_normalize_data method start")
        columns = data.columns
        data = np.asarray(data, dtype=np.float32)
        if np.any(sum(np.isnan(data))):
            data = np.nan_to_num(data)

        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data, scaler, columns

    def _save(self, group_model_tuple_list):
        if os.path.exists(self.model_dir):
            pass
        else:
            os.mkdir(self.model_dir)
        for file_name, model, scaler, columns, threshold, anomaly_cdf in group_model_tuple_list:
            # 모델 저장
            model_file_name = file_name + ".onnx"
            model_PATH = self.model_dir + "/" + model_file_name
            dummy_input = torch.randn(1, self.window_size, len(columns)).to(self.device)
            input_names = ["input"]
            output_names = ["recon", "pred"]
            ONNXTorch.onnx_save(model, dummy_input, model_PATH, input_names=input_names,
                                output_names=output_names, opset_version=10)
            # compose bundle of scaler, column order, threshold
            bundle = {"scaler": scaler, "columns": list(columns), "threshold": threshold, "anomaly_cdf": anomaly_cdf}
            bundle_file_name = file_name + "_bundle.pkl"
            bundle_PATH = self.model_dir + "/" + bundle_file_name
            with open(str(bundle_PATH), "wb") as pickle_file:
                pickle.dump(bundle, pickle_file)

    def init_param(self, config):
        pass

    def init_serve(self, reload=False):
        pass

    def _load(self, model_name):
        pass

    def serve(self, header, input_df):
        pass

    def end_serve(self):
        pass

    def get_debug_info(self):
        pass

