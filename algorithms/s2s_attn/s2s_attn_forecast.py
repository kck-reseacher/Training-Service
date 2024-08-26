from algorithms.s2s_attn.s2s_attn import Seq2seqAttention

import numpy as np

from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, RepeatVector, TimeDistributed, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from common import constants


class S2SAttnForecast(Seq2seqAttention):
    def __init__(self, id, config, logger):
        super().__init__(id, config, logger)

        self.set_vars(constants.MODEL_S_S2S)
        self.init_config(config)

        self.algo_log = 'S2S_Attn'
        self.model_desc = 's2s_attn'
        self.progress_name = 'Seq2Seq-Attention'
        self.drop_rate = {'pred': 0.1, 'band': 0.25}

        self.pred_horizon = 30

    def init_param(self, config):
        super().init_param(config)
        self.attn_out = dict([(feat, hidden_size if hidden_size is None else hidden_size // 2) for feat, hidden_size in self.lstm_out.items()])

    def get_model(self, feat, **kwargs):
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

    def get_sequence_data(self, feat, feat_data, for_training=True):
        x_data, y_data = [], []
        for i in range(feat_data.shape[0] - self.window_size[feat] - self.pred_horizon + 1):
            x_data.append(feat_data[i: i+self.window_size[feat]].reshape(-1, 1))
            if for_training:
                y_data.append(feat_data[i+self.window_size[feat]: i+self.window_size[feat]+self.pred_horizon].reshape(-1, 1))

        return np.array(x_data), np.array(y_data) if for_training else np.array(x_data)
