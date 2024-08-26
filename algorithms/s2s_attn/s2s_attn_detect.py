from algorithms.s2s_attn.s2s_attn import Seq2seqAttention

import numpy as np

from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, RepeatVector, TimeDistributed, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from common import constants


class S2SAttnDetect(Seq2seqAttention):
    def __init__(self, id, config, logger):
        super().__init__(id, config, logger)

        self.set_vars(constants.MODEL_S_SEQATTN)
        self.init_config(config)

        self.algo_log = 'SeqAttn'
        self.model_desc = 'seqattn'
        self.progress_name = 'SeqAttn'
        self.drop_rate = {'pred': 0.0, 'band': 0.3}

    def init_config(self, config):
        super().init_config(config)

        if config['parameter'].get('service', None) is not None:
            for feat, sigma_val in config['parameter']['service'][self.algo_str]['range'].items():
                self.sigma_coef[feat] = sigma_val

    def init_param(self, config):
        super().init_param(config)
        self.attn_out = self.get_hyper_param_values(self.params_s2s, 'attention_size')

    def update_band_width(self, parameter):
        range_dict = parameter['parameter']['service'][constants.MODEL_S_SEQATTN]['range']
        for feat in self.pred_feats:
            if feat not in range_dict.keys():
                self.logger.info(f"{feat} not found in range_dict received from server")
            else:
                prev_val = self.sigma_coef[feat]
                self.sigma_coef[feat] = range_dict[feat]
                self.logger.info(f"{feat} sigma_coef changed : {prev_val} => {self.sigma_coef[feat]}")

    def is_line_patterned(self, feat_data):
        feat_ma = np.convolve(feat_data, np.ones(10), mode='valid') / 10
        feat_vals = feat_data[-feat_ma.shape[0]:]
        return True if feat_ma[feat_ma == feat_vals].shape[0] / feat_ma.shape[0] >= 0.9 else False

    def get_model(self, feat, **kwargs):
        is_line_patterned = kwargs.get('is_line_patterned', True)
        enc_inp = Input(shape=(self.window_size[feat], 1))

        if not is_line_patterned:
            inp_vec = Dense(1, kernel_initializer='uniform')(enc_inp)
            inp_wave = K.sin(Dense(256, kernel_initializer='uniform')(enc_inp))
            enc_t2v = Concatenate(axis=-1)([inp_vec, inp_wave])

        hidden_enc, encoder_h, encoder_c = LSTM(self.lstm_out[feat], return_sequences=True, return_state=True)(enc_t2v if not is_line_patterned else enc_inp)
        dec_inp = RepeatVector(self.window_size[feat])(encoder_h)
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

    def get_sequence_data(self, feat, feat_data, for_training=True):
        x_data, y_data = [], []
        for i in range(feat_data.shape[0] - self.window_size[feat] + 1):
            x_data.append(feat_data[i: i+self.window_size[feat]].reshape(-1, 1))
        if for_training:
            y_data = list(map(lambda x: x[::-1], x_data))

        return np.array(x_data), np.array(y_data) if for_training else np.array(x_data)