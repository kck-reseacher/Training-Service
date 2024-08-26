import datetime
import json
import os
import shutil
from pathlib import Path

import joblib

from algorithms import aimodel

from abc import *

import numpy as np
import pandas as pd
from copy import deepcopy

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.losses import MSE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import exceptions

import time

from common.aicommon import JsonEncoder, Query, Utils, sMAPE
from common.error_code import Errors
from common.timelogger import TimeLogger
from common import constants

from common.memory_analyzer import MemoryUtil
from common.module_exception import ModuleException
from common.onnx import ONNX
from common.redisai import REDISAI


class Seq2seqAttention(aimodel.AIModel, metaclass=ABCMeta):
    def __init__(self, id, config, logger):
        self.mu = MemoryUtil(logger)

        self.logger = logger
        self.config = config
        self.model_dir = self.config['model_dir']

        self.business_list = config.get('business_list', None)
        self.except_failure_date_list = config.get('except_failure_date_list', [])
        self.except_business_list = config.get('except_business_list', [])

        self.algo_str = None
        self.params_s2s = None
        self.pred_feats = None

        self.algo_log = None
        self.model_desc = None
        self.progress_name = None
        self.drop_rate = {}
        self.model_id = None

        self.test_perc = None

        self.scalers = {}
        self.models = {}

        # model params
        self.params_s2s = None
        self.window_size = {}
        self.lstm_out = {}
        self.attn_out = {}
        self.batch_size = {}
        self.lr = {}
        self.epochs = {}

        # band params (const)
        self.n_band_iter = 50

    def set_vars(self, algo_str):
        self.algo_str = algo_str
        self.params_s2s = self.config['parameter']['train'][self.algo_str]
        self.pred_feats = self.params_s2s['features']
        self.sigma_coef = {feat: 3 for feat in self.pred_feats}

    def init_config(self, config):
        self.config = config

        self.params_s2s = config['parameter']['train'][self.algo_str]
        self.pred_feats = self.params_s2s['features']
        self.test_perc = config['parameter']['data_set']['test'] if 'data_set' in config['parameter'].keys() else None

        self.init_param(config)

        self.scalers = dict([(feat, StandardScaler()) for feat in self.pred_feats])
        self.set_vars(self.algo_str)

    def get_hyper_param_values(self, params, hyper_param):
        if 'metric_hyper_params' in params.keys():
            metrics = params["metric_hyper_params"].keys()
            return dict([(metric, params["metric_hyper_params"][metric].get(hyper_param, None)) for metric in metrics])
        else:
            return dict()

    def init_param(self, config):
        self.params_s2s = config['parameter']['train'][self.algo_str]
        self.pred_feats = self.params_s2s['features']
        self.window_size = self.get_hyper_param_values(self.params_s2s, 'window_size')
        self.lstm_out = self.get_hyper_param_values(self.params_s2s, 'hidden_unit_size')
        self.batch_size = self.get_hyper_param_values(self.params_s2s, 'batch_size')
        self.lr = self.get_hyper_param_values(self.params_s2s, 'learning_rate')
        self.epochs = self.get_hyper_param_values(self.params_s2s, 'epochs')

    def is_line_patterned(self, feat_data):
        return None

    @abstractmethod
    def get_model(self, feat, **kwargs):
        pass

    @abstractmethod
    def get_sequence_data(self, feat, feat_data, for_training=True):
        pass

    def remove_global_outliers(self, feat_data):
        feat_data = np.log1p(feat_data.astype(np.float))
        mean_, std_ = feat_data.mean(), feat_data.std()

        outlier_idx = np.where((feat_data < (mean_ - 3 * std_)) | (feat_data > (mean_ + 3 * std_)))[0]
        feat_data[outlier_idx] = np.nan
        feat_series = pd.Series(feat_data)
        self.logger.info(f"n_global_outliers = {outlier_idx.shape[0]}")

        feat_series.interpolate(limit_area='inside', inplace=True)
        feat_series.interpolate(limit_direction='both', inplace=True)

        feat_data = np.expm1(feat_series.values)
        return feat_data

    def fit(self, df, biz_data_dict={}, train_progress=None):
        if len(df) == 0:
            return None, None, Errors.E800.value, Errors.E800.desc
        elif df.shape[0] < constants.DBSLN_DMIN_MAX * 2:
            return None, None, Errors.E801.value, Errors.E801.desc
        elif len(self.pred_feats) == 0:
            self.logger.error("Y feature does not exist")
            return None, None, Errors.E802.value, Errors.E802.desc

        ## data preprocessing ##
        if biz_data_dict is None:
            biz_data_dict = {}

        failure_dates, except_biz_dates = [], []
        if len(self.except_failure_date_list) != 0:
            df, failure_dates = Utils.drop_failure_date(self.except_failure_date_list, df)
            for biz_idx in biz_data_dict.keys():
                biz_data_dict[biz_idx], _ = Utils.drop_failure_date(self.except_failure_date_list, biz_data_dict[biz_idx])

        if len(self.except_business_list) != 0:
            df, except_biz_dates = Utils.drop_except_business_list(self.except_business_list, df)
            for biz_idx in biz_data_dict.keys():
                if biz_data_dict[biz_idx].shape[0] > 0:
                    biz_data_dict[biz_idx], _ = Utils.drop_except_business_list(self.except_business_list, biz_data_dict[biz_idx])
                else:
                    biz_data_dict[biz_idx] = pd.DataFrame()

        for i in range(len(self.business_list)):
            biz_idx = self.business_list[i]['index']
            self.business_list[i]['result'] = 1 if len(biz_data_dict.get(biz_idx, [])) > 0 else -1
        ## data preprocessing ##

        with TimeLogger(f"[{self.algo_log}] model training time :", self.logger):
            total_start = time.time()

            total_biz_df = pd.concat([biz_data for biz_idx, biz_data in biz_data_dict.items()]) if bool(biz_data_dict) else None

            feats_metrics_result = {}

            trained_feats = deepcopy(self.pred_feats)
            for feat in self.pred_feats:
                self.logger.info(f"feature = {feat}({self.pred_feats.index(feat) + 1}/{len(self.pred_feats)})")
                feat_start = time.time()
                if feat not in df.columns:
                    self.logger.info(f"{Errors.E804.desc}")
                    Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress,
                                                              self.progress_name, int((self.pred_feats.index(feat) + 1) / len(self.pred_feats) * 100))
                    trained_feats.remove(feat)
                    continue

                feat_data = self.remove_global_outliers(df[feat].dropna().values)
                _, feat_test = train_test_split(feat_data, test_size=self.test_perc / 100, shuffle=False)

                # merge bizday data into non-bizday data
                train_business_status = None
                self.logger.info(f"df.shape = {df.shape}")
                if total_biz_df is not None:
                    self.logger.info(f"total_biz_df.shape = {total_biz_df.shape}")
                    feat_data_biz = self.remove_global_outliers(total_biz_df[feat].dropna().values)
                    _, feat_test_biz = train_test_split(feat_data_biz, test_size=self.test_perc / 100, shuffle=False)
                    feat_data = np.append(feat_data, feat_data_biz)
                    train_business_status = {str(biz_idx): True for biz_idx in biz_data_dict.keys()}

                # scale
                feat_data_scaled = self.scalers[feat].fit_transform(feat_data.reshape(-1, 1)).reshape(-1)
                feat_test_scaled = self.scalers[feat].transform(feat_test.reshape(-1, 1)).reshape(-1)

                # sequence
                x_train, y_train = self.get_sequence_data(feat, feat_data_scaled)
                x_test, y_test = self.get_sequence_data(feat, feat_test_scaled)

                # model
                self.models[feat] = self.get_model(feat, is_line_patterned=self.is_line_patterned(feat_data))
                callbacks = [LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(f"[{self.algo_log}] epoch = {epoch + 1}/{self.epochs[feat]}, loss = {logs['loss']:.4f}, val_loss = {logs['val_loss']:.4f}"))]

                ## training (heuristic) ##
                fit_start = time.time()

                init_epoch = 0
                almost_zero_in_data = bool((df[df[feat] == 0].shape[0] / df.shape[0]) > 0.89)
                if not almost_zero_in_data:
                    init_epoch = 20
                    self.models[feat].fit(x_train, y_train, batch_size=self.batch_size[feat], epochs=init_epoch,
                                          validation_data=(x_test, y_test), callbacks=callbacks, verbose=0)

                callbacks.append(EarlyStopping(monitor='loss', min_delta=0.001, patience=5, restore_best_weights=True))
                self.models[feat].fit(x_train, y_train, batch_size=self.batch_size[feat], epochs=self.epochs[feat], initial_epoch=init_epoch,
                                      validation_data=(x_test, y_test), callbacks=callbacks, verbose=0)

                fit_end = time.time()
                self.logger.info(f"Training {feat} model elapsed = {fit_end - fit_start:.3f}s")
                ## training ##

                # calc metrics
                inv_scaled = lambda x: self.scalers[feat].inverse_transform(x.reshape(-1, 1)).reshape(-1)
                pred_test = self.models[feat].predict(x_test, batch_size=self.batch_size[feat])
                smape = sMAPE(inv_scaled(y_test), inv_scaled(pred_test))
                mse = MSE(inv_scaled(y_test), inv_scaled(pred_test)).numpy()

                feat_end = time.time()

                Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress,
                                                          self.progress_name, int((self.pred_feats.index(feat) + 1) / len(self.pred_feats) * 100))
                feats_metrics_result[feat] = {'mse': mse, 'rmse': np.sqrt(mse), 'smape': smape,
                                              'duration_time': feat_end - feat_start, 'hyper_params': self.params_s2s['metric_hyper_params'][feat]}

            total_end = time.time()
            Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, self.progress_name, 100)

        self.pred_feats = trained_feats

        from_date, to_date = pd.to_datetime(df.index[[0, -1]], format=constants.INPUT_DATETIME_FORMAT).map(lambda x: x.strftime(constants.INPUT_DATE_FORMAT))

        duration_time = round(total_end - total_start)
        Query.update_module_status_by_training_id(self.config['db_conn_str'],
                                                  self.config['train_history_id'],
                                                  train_progress, self.progress_name, 100,
                                                  duration_time)

        train_result = {'from_date': from_date, 'to_date': to_date,
                        'results': {'mse': -1, 'rmse': -1, 'duration_time': duration_time, 'hyper_params': None},
                        'train_metrics': feats_metrics_result, 'except_failure_date_list': failure_dates, "except_business_list": except_biz_dates,
                        "business_list": self.business_list, "train_business_status": train_business_status, "train_mode": -1, "outlier_mode": -1}
        return train_result, None, 0, None

    def save_files(self, model_dir, feat):
        feat_model_path = os.path.join(model_dir, f"{self.model_desc}_{feat}.h5")
        feat_scaler_path = os.path.join(model_dir, f"{self.model_desc}_{feat}_scaler.pkl")
        onnx_feat_model_path = os.path.join(model_dir, f"{self.model_desc}_{feat}.onnx")

        # model
        for dropout_layer in list(filter(lambda layer: isinstance(layer, Dropout), self.models[feat].layers)):
            dropout_layer.rate = self.drop_rate['band']

        # Convert Tensorflow to ONNX
        ONNX.onnx_save(self.models[feat], onnx_feat_model_path)

        # scaler
        joblib.dump(self.scalers[feat], feat_scaler_path)

        self.logger.info(f"{feat} model/scaler have been saved to {model_dir}")

    def save(self, path):
        path = str(Path(path) / self.algo_str)

        if os.path.exists(path):
            shutil.rmtree(path)

        Path(path).mkdir(exist_ok=True, parents=True)

        for feat in self.models.keys():
            self.save_files(path, feat)
