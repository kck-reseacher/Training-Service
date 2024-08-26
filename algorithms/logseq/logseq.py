import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, MultiHeadAttention
from tensorflow.keras.metrics import TopKCategoricalAccuracy, top_k_categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence

from algorithms import aimodel
from algorithms.logseq.log2template import Log2Template
from common import constants, aicommon
from common.error_code import Errors
from common.memory_analyzer import MemoryUtil
from common.onnx import ONNX
from common.timelogger import TimeLogger


################################################################
# template_index range (1 ~ n) => softmax_index range (0 ~ n-1)
################################################################

class LogDataGenerator(Sequence):
    def __init__(self, x_data, y_data, batch_size=1024):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_data))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return round(len(self.x_data) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size: (idx+1)*self.batch_size]
        x_batch = self.x_data[indexes]
        y_batch = self.y_data[indexes]

        return x_batch, y_batch


class LogSeq(aimodel.AIModel):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.logseq_params = None
        self.window_size = 30
        self.batch_size = 1024
        self.epochs = 50
        self.hidden_size = 32
        self.test_perc = 30

        self.top_k = constants.DEFAULT_N_TOP_K
        self.anomaly_threshold = None

        # template model
        self.log2template = Log2Template(config, logger)
        self.most_freq_idxs = []

        # tf model
        self.model = None
        self.model_dir = None

        self.progress_name = constants.MODEL_B_LOGSEQ
        self.under_number_of_digits = 6#소수점 버림 자릿수

        self.service_id = constants.MODEL_S_LOGSEQ

        self.mu = MemoryUtil(logger)

    def init_train(self):
        self.logseq_params = self.config['parameter']['train'][constants.MODEL_S_LOGSEQ]['metric_hyper_params'][constants.MSG]
        self.window_size = self.logseq_params['window_size']
        self.batch_size = self.logseq_params['batch_size']
        self.epochs = self.logseq_params['epochs']
        self.hidden_size = self.logseq_params['hidden_unit_size']

        self.test_perc = self.config['parameter']['data_set']['test'] if 'data_set' in self.config['parameter'].keys() else None

    def get_model(self):
        n_classes = self.log2template.n_templates
        embed_size = 128

        inp = Input(shape=(self.window_size,))
        emb = Embedding(n_classes+1, embed_size)(inp)
        lstm, state_h, _ = LSTM(self.hidden_size, return_sequences=True, return_state=True)(emb)
        attn = MultiHeadAttention(num_heads=10, key_dim=64)(state_h[:, tf.newaxis, :], lstm)
        attn = K.squeeze(attn, axis=1)
        out = Dense(n_classes, activation='softmax')(attn)
        model = Model(inp, out)
        model.compile(optimizer=Adam(learning_rate=0.005), loss='categorical_crossentropy',
                      metrics=[TopKCategoricalAccuracy(k, name=f"top_{k}") for k in [1, 3, 5, 10, 20]])

        return model

    def get_sequence_data(self, data):
        x_data, y_data = [], []
        for i in range(len(data) - self.window_size):
            x_data.append(data[i:i + self.window_size])
            y_data.append(data[i + self.window_size])

        x_data, y_data = np.array(x_data), to_categorical(y_data, num_classes=self.log2template.n_templates)
        return x_data, y_data

    def fit(self, log_df, train_progress=None):
        if not os.path.exists(os.path.join(self.config['model_dir'], f"{constants.MODEL_S_LOGSEQ}")):
            os.makedirs(os.path.join(self.config['model_dir'], f"{constants.MODEL_S_LOGSEQ}"))

        if log_df.shape[0] < 60+1:  # max window_size + 1
            return None, None, Errors.E801.value, Errors.E801.desc

        with TimeLogger(f"[LogSeq] model training time :", self.logger):
            time_fit_s = time.time()

            log_df['msg'] = log_df['msg'].apply(lambda x: x.replace('"', '""'))

            log_df = self.log2template.log2tidx(log_df, fitting=True, train_progress=train_progress)

            x_data, y_data = self.get_sequence_data(log_df["tidx"])

            _, x_test = train_test_split(x_data, test_size=self.test_perc / 100, shuffle=False)
            _, y_test = train_test_split(y_data, test_size=self.test_perc / 100, shuffle=False)

            train_data_generator = LogDataGenerator(x_data, y_data, min(x_data.shape[0], self.batch_size))
            test_data_generator = LogDataGenerator(x_test, y_test, min(x_test.shape[0], self.batch_size))

            # Model
            self.logger.info(f"[LogSeq] tf_model training start")
            time_model_s = time.time()
            self.model = self.get_model()
            callbacks = [LoggerCallback(self.epochs, self.logger, self.progress_name, self.config, train_progress),
                         EarlyStopping(monitor='val_top_5', min_delta=0.01, mode='max', patience=3, restore_best_weights=True)]
            hist = self.model.fit_generator(train_data_generator, epochs=self.epochs, validation_data=test_data_generator, callbacks=callbacks, verbose=0)
            time_model_e = time.time()
            self.logger.info(f"[LogSeq] tf_model training end (elapsed = {time_model_e - time_model_s:.3f}s)")

            aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, self.progress_name, 90)

            self.mu.print_memory()

            y_pred = self.model.predict(x_test, batch_size=1024)
            serving_top_k = 1
            for serving_top_k in range(1, self.log2template.n_templates):
                val_acc = top_k_categorical_accuracy(y_test, y_pred, serving_top_k).numpy().mean()
                self.logger.info(f"[LogSeq] serving_top_k = {serving_top_k}, val_acc = {val_acc:.3f}")
                if val_acc >= 0.95:
                    break

            df_test = log_df.iloc[-y_test.shape[0]:]
            df_test['time'] = pd.to_datetime(df_test['time'].values).strftime('%Y%m%d%H%M')
            size_1m = df_test[['time', 'tidx']].groupby('time').agg('count').values.squeeze()
            n_anomalies_1m = []
            idx_from = 0
            for idx_to in size_1m:
                res = top_k_categorical_accuracy(y_test[idx_from: idx_from+idx_to], y_pred[idx_from: idx_from+idx_to], serving_top_k).numpy()
                n_anomalies_1m.append(res.shape[0] - res.sum())
                idx_from += idx_to
            serving_anomaly_threshold = max(1, np.round(np.mean(n_anomalies_1m)).astype(int))
            self.logger.info(f"[LogSeq] serving_anomaly_threshold = {serving_anomaly_threshold}")


            time_fit_f = time.time()
            duration_time = round(time_fit_f - time_fit_s)

            aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, self.progress_name, 100, duration_time)

            msg_metric_result = {
                "rmse": {k: aicommon.Utils.decimal_point_discard(v[-1], self.under_number_of_digits) for k, v in hist.history.items() if "val_top" in k},
                "deploy_policy": -1,
                "hyper_params": self.logseq_params,
            }

            report = {
                "duration_time": duration_time,
                "mse": -1,
                "rmse": -1,
                "pre_mse": -1,
                "pre_rmse": -1,
                "deploy_policy": -1,
            }
            train_metrics = {
                constants.MSG: msg_metric_result
            }

            # TODO - return template_clusters
            train_result = {"from_date": self.config['date'][0],
                            "to_date": self.config['date'][-1],
                            "accuracy": {k: aicommon.Utils.decimal_point_discard(v[-1], self.under_number_of_digits) for k, v in hist.history.items()},
                            "train_metrics": train_metrics,
                            'mined_period': self.log2template.mined_period,
                            # 'n_templates': self.log2template.n_templates,
                            # 'templates': [c.get_template() for c in self.log2template.template_miner.drain.clusters],
                            'serving_top_k': serving_top_k,
                            'anomaly_threshold': serving_anomaly_threshold,
                            "except_failure_date_list": None,
                            "except_business_list": None,
                            "business_list": None,
                            "train_business_status": None,
                            "train_mode": -1,
                            "outlier_mode": -1,
                            "results": report
                            }

            return train_result

    def save(self, model_dir):
        self.log2template.save(model_dir)

        model_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/tf_model.h5")
        self.model.save(model_path)
        try:
            onnx_model_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/onnx_model.onnx")
            # tf.int64 받을 수 있는 ONNX 모델로 convert
            input_signature = [tf.TensorSpec([None, 30], tf.int64)]
            ONNX.onnx_save(self.model, onnx_model_path, input_signature=input_signature)
        except:
            pass

        self.logger.info(f"[LogSeq] tf_model saved (3/4)")

        etc_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/etc_info.pkl")
        etc_info = {'top_k': self.top_k, 'mined_period': self.log2template.mined_period}
        joblib.dump(etc_info, etc_path)
        self.logger.info(f"[LogSeq] etc_info saved (4/4)")


class LoggerCallback(Callback):
    def __init__(self, epochs, logger, progress_name, config, train_progress):
        super().__init__()
        self.logger = logger
        self.epochs = epochs
        self.progress_name = progress_name
        self.config = config
        self.train_progress = train_progress

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(f"[LogSeq] epoch={epoch + 1}/{self.epochs}"
                         f", loss=(train={logs['loss']:.3f}/test={logs['val_loss']:.3f})"
                         f", top_1=({logs['top_1']:.3f}/{logs['val_top_1']:.3f})"
                         f", top_3=({logs['top_3']:.3f}/{logs['val_top_3']:.3f})"
                         f", top_5=({logs['top_5']:.3f}/{logs['val_top_5']:.3f})"
                         f", top_10=({logs['top_10']:.3f}/{logs['val_top_10']:.3f})"
                         f", top_20=({logs['top_20']:.3f}/{logs['val_top_20']:.3f})")

        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], self.train_progress,
                                                           self.progress_name, 30 + int((epoch + 1) / self.epochs * 40))  # 30 ~ 70%
