import time
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, RepeatVector, TimeDistributed, Dropout, \
    MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pathos
import psutil
from algorithms import aimodel
from common.aicommon import Query
from common.error_code import Errors
from common import constants
from common.timelogger import TimeLogger
from common.onnx import ONNX
from pathlib import Path
from common.aicommon import sMAPE

class Config:
    batch_size = 2048
    epoch = 50
    dim = 64
    learning_rate = 0.005
    window_size = 30
    pred_horizon = 30
    drop_rate = {'pred': 0.1, 'band': 0.1}
    test_proportion = 30


class LoadForecast(aimodel.AIModel):
    def __init__(self, id, config, logger):

        self.logger = logger
        self.config = config

        self.algo_log = 'S2S_Attn'
        self.model_desc = 's2s_attn'
        self.progress_name = 'Seq2Seq-Attention'
        self.window_size = {}

        self.drop_rate = Config.drop_rate
        self.pred_horizon = Config.pred_horizon

        self.multi_scalers = {}
        self.models = {}

        self.is_multiprocessing_mode = True
        self.is_multithread_mode = True
        self.number_of_child_processes = int(psutil.cpu_count(logical=True) * 0.3)

    def init_config(self, config):
        self.config = config
        self.model_dir = self.config['model_dir']

        parameters = config['parameter']['train']["seq2seq"]
        self.pred_feats = parameters['features']
        self.test_ratio = Config.test_proportion
        self.sigma_coef = dict([(feat, 3) for feat in self.pred_feats])
        self.init_param()

    def init_param(self):
        self.window_size = dict([(feat, Config.window_size) for feat in self.pred_feats])
        self.epochs = dict([(feat, Config.epoch) for feat in self.pred_feats])
        self.batch_size = dict([(feat, Config.batch_size) for feat in self.pred_feats])
        self.lr = dict([(feat, Config.learning_rate) for feat in self.pred_feats])
        self.lstm_out = dict([(feat, Config.dim) for feat in self.pred_feats])
        self.attn_out = dict([(feat, hidden_size if hidden_size is None else hidden_size // 2) for feat, hidden_size in
                              self.lstm_out.items()])

    def get_model(self, feat):
        enc_inp = Input(shape=(self.window_size[feat], 1))
        hidden_enc, encoder_h, encoder_c = LSTM(self.lstm_out[feat], return_sequences=True, return_state=True)(enc_inp)
        dec_inp = RepeatVector(self.pred_horizon)(encoder_h)
        hidden_dec = LSTM(self.lstm_out[feat], return_sequences=True)(dec_inp, initial_state=[encoder_h, encoder_c])
        hidden_dec = Dropout(0)(hidden_dec, training=True)
        hidden = MultiHeadAttention(num_heads=1, key_dim=self.lstm_out[feat])(hidden_enc, hidden_dec)
        hidden = Concatenate()([hidden, hidden_dec])
        hidden = TimeDistributed(Dense(self.attn_out[feat], use_bias=False, activation='tanh'))(hidden)
        hidden = Dropout(0)(hidden, training=True)
        dec_out = Dense(1)(hidden)

        model = Model(enc_inp, dec_out)
        model.compile(optimizer=Adam(self.lr[feat]), loss='mse')

        return model

    def remove_global_outliers(self, feat_data):
        feat_data = np.log1p(feat_data.astype(float))
        mean_, std_ = feat_data.mean(), feat_data.std()

        outlier_idx = np.where((feat_data < (mean_ - 3 * std_)) | (feat_data > (mean_ + 3 * std_)))[0]
        feat_data[outlier_idx] = np.nan
        feat_series = pd.Series(feat_data)
        self.logger.info(f"n_global_outliers = {outlier_idx.shape[0]}")

        feat_series.interpolate(limit_area='inside', inplace=True)
        feat_series.interpolate(limit_direction='both', inplace=True)

        feat_data = np.expm1(feat_series.values)
        return feat_data

    def sequence_data_generator(self, batch_size, feat, feat_data):
        batch_iter_size = batch_size
        for idx, i in enumerate(range(0, feat_data.shape[0], batch_size)):
            x_batch, y_batch = [], []
            if idx >= (feat_data.shape[0] // batch_size) - 1:
                batch_iter_size = feat_data.shape[0] - i - self.window_size[feat] - self.pred_horizon + 1
            if i > feat_data.shape[0] - self.window_size[feat] - self.pred_horizon:
                continue
            for j in range(batch_iter_size):
                start_idx = i + j
                end_idx = start_idx + self.window_size[feat]
                x_batch.append(feat_data[start_idx: end_idx].reshape(-1, 1))

                start_idx = end_idx
                end_idx = start_idx + self.pred_horizon
                y_batch.append(feat_data[start_idx: end_idx].reshape(-1, 1))
            yield np.array(x_batch), np.array(y_batch)

    def remove_outlier_func(self, target_df, target_id):
        scalers = {}
        scaled_data = {target_id: {}}
        scaler_result = {}
        for feat in self.pred_feats:
            if feat in target_df.columns:
                if target_df[feat].isnull().sum() != len(target_df):
                    scalers[feat] = StandardScaler()
                    target_df[feat][target_df[feat] < 0] = np.nan
                    target_df[feat] = target_df[feat].interpolate(method='linear').fillna(method='ffill').fillna(
                        method='bfill').fillna(0)
                    feat_data = self.remove_global_outliers(target_df[feat].dropna().values)
                    feat_data_scaled = scalers[feat].fit_transform(feat_data.reshape(-1, 1)).reshape(-1)
                    scaled_data[target_id][feat] = feat_data_scaled
        scaler_result[target_id] = scalers

        return (scaled_data, scaler_result)

    def fit(self, df, train_progress=None):

        total_start = time.time()

        if self.is_multiprocessing_mode:
            pool = pathos.multiprocessing.Pool(processes=self.number_of_child_processes)
            input_iterable = [(df.loc[df["target_id"] == target_id], target_id) for target_id in
                              np.unique(df["target_id"].values)]
            chunk_size, remainder = divmod(
                len(input_iterable), self.number_of_child_processes
            )
            if remainder != 0:
                chunk_size += 1

            scaled_dataset = list()
            scaler_results = list()
            for result in pool.starmap(self.remove_outlier_func, input_iterable, chunksize=chunk_size):
                scaled_dataset.append(result[0])
                scaler_results.append(result[1])
            pool.close()
            pool.join()
            for element in scaled_dataset:
                if element is None:
                    continue
                else:
                    for target_id in list(element.keys()):
                        for feat in list(element[target_id].keys()):
                            df.loc[df['target_id'] == target_id, feat] = element[target_id][feat]

            for target_scaler in scaler_results:
                if target_scaler is None:
                    continue
                else:
                    for target_id in list(target_scaler.keys()):
                        self.multi_scalers[str(target_id)] = target_scaler[target_id]
        else:
            # multi target scale
            df["target_id"] = df["target_id"].astype(str)
            for target_id in np.unique(df["target_id"].values):
                scalers = {}
                target_df = df.loc[df["target_id"] == target_id]
                self.logger.info(f"target_df : {target_df}")
                for feat in self.pred_feats:
                    if feat in target_df.columns:
                        if target_df[feat].isnull().sum() != len(target_df):
                            scalers[feat] = StandardScaler()
                            target_df[feat][target_df[feat] < 0] = np.nan
                            target_df[feat] = target_df[feat].interpolate(method='linear').fillna(
                                method='ffill').fillna(method='bfill').fillna(0)
                            feat_data = self.remove_global_outliers(target_df[feat].dropna().values)
                            feat_data_scaled = scalers[feat].fit_transform(feat_data.reshape(-1, 1)).reshape(-1)
                            df.loc[df['target_id'] == target_id, feat] = feat_data_scaled
                self.multi_scalers[str(target_id)] = scalers

        self.logger.info(f"df : {df}")

        with TimeLogger(f"[{self.algo_log}] model training time :", self.logger):
            feats_metrics_result = dict([(metric, None) for metric in self.pred_feats])
            for feat in self.pred_feats:
                self.logger.info(f"{feat} model training start !!")
                fit_start = time.time()
                if feat not in df.columns:
                    self.logger.info(f"{Errors.E804.desc}")
                    Query.update_module_status_by_training_id(self.config['db_conn_str'],
                                                              self.config['train_history_id'], train_progress,
                                                              self.progress_name,
                                                              int((self.pred_feats.index(feat) + 1) / len(
                                                                  self.pred_feats) * 100))
                    continue

                init_epoch = 0
                model = self.get_model(feat)
                feat_train = df[feat].dropna().values

                test_dataset = []
                for target_id in np.unique(df["target_id"].values):
                    feat_data = df.loc[df['target_id'] == target_id, feat].dropna().values
                    if len(feat_data) != 0:
                        _, test = train_test_split(feat_data, test_size=self.test_ratio / 100, shuffle=False)
                        test_dataset.append(test)
                feat_test = np.concatenate(test_dataset)

                train_dataset = tf.data.Dataset.from_generator(
                    lambda: self.sequence_data_generator(self.batch_size[feat], feat, feat_train),
                    output_signature=(
                        tf.TensorSpec(shape=(None, self.window_size[feat], 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, self.pred_horizon, 1), dtype=tf.float32)
                    )
                ).shuffle(buffer_size=round(len(feat_train) * 0.1))

                valid_dataset = tf.data.Dataset.from_generator(
                    lambda: self.sequence_data_generator(self.batch_size[feat], feat, feat_test),
                    output_signature=(
                        tf.TensorSpec(shape=(None, self.window_size[feat], 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, self.pred_horizon, 1), dtype=tf.float32)
                    )
                ).shuffle(buffer_size=round(len(feat_test) * 0.1))

                callbacks = [LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(
                    f"[{self.algo_log}] epoch = {epoch + 1}/{self.epochs[feat]}, loss = {logs['loss']:.4f}, val_loss = {logs['val_loss']:.4f}"))]

                almost_zero_in_data = bool((df[df[feat] == 0].shape[0] / df.shape[0]) > 0.89)
                if not almost_zero_in_data:
                    init_epoch = 20
                    model.fit(train_dataset, batch_size=self.batch_size[feat], epochs=init_epoch,
                                          validation_data=valid_dataset, callbacks=callbacks, verbose=0)

                callbacks.append(EarlyStopping(monitor='loss', min_delta=0.001, patience=5, restore_best_weights=True))
                model.fit(train_dataset, epochs=self.epochs[feat], initial_epoch=init_epoch,
                                      validation_data=valid_dataset, callbacks=callbacks, verbose=0)
                true_vals = np.concatenate([y for x, y in valid_dataset], axis=0)
                pred_vals = model.predict(valid_dataset)
                smape_score = np.round(sMAPE(true_vals, pred_vals), 2)
                self.logger.info(f"feat: {feat}, sMAPE : {smape_score}")
                fit_end = time.time()
                self.logger.info(f"Training {feat} model elapsed = {fit_end - fit_start:.3f}s")
                Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'],
                                                          train_progress,
                                                          self.progress_name,
                                                          int((self.pred_feats.index(feat) + 1) / len(
                                                              self.pred_feats) * 100))
                feats_metrics_result[feat] = {'smape': smape_score,
                                              'duration_time': fit_end - fit_start, 'hyper_params':
                                                  self.config['parameter']['train']["seq2seq"]['metric_hyper_params'][
                                                      feat]}
                # model save
                self.save_files(self.model_dir, model, feat)

            from_date, to_date = pd.to_datetime(df.index[[0, -1]], format=constants.INPUT_DATETIME_FORMAT).map(
                lambda x: x.strftime(constants.INPUT_DATE_FORMAT))

            total_end = time.time()
        duration_time = round(total_end - total_start)
        untrained_target_list = list(set(self.config.get("target_list", [])) - set(df["target_id"].values))
        Query.update_module_status_by_training_id(self.config['db_conn_str'],
                                                  self.config['train_history_id'],
                                                  train_progress, self.progress_name, 100,
                                                  duration_time)
        train_result = {"from_date": from_date, "to_date": to_date,
                        "results": {"mse": -1, "rmse": -1, "duration_time": duration_time, "hyper_params": None},
                        "target_list": list(np.unique(df["target_id"].values)),
                        "train_metrics": feats_metrics_result, "except_failure_date_list": [],
                        "except_business_list": [],
                        "business_list": [], "train_business_status": [], "train_mode": -1, "outlier_mode": -1,
                        "untrained_target_list" : untrained_target_list}
        self.save_scaler(self.model_dir)

        return train_result, None, 0, None

    def save_files(self, model_dir, model, feat):
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        onnx_feat_model_path = os.path.join(model_dir, f"{self.model_desc}_{feat}.onnx")

        # Convert Tensorflow to ONNX
        ONNX.onnx_save(model, onnx_feat_model_path)
        self.logger.info(f"{onnx_feat_model_path} save success")

    def save_scaler(self, model_dir):
        scaler_path = os.path.join(model_dir, f"{self.model_desc}_scaler.pkl")
        joblib.dump(self.multi_scalers, scaler_path)
        self.logger.info(f"scaler save success")