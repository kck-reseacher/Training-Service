import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class TSFDataLoader:

  def __init__(self, data, batch_size, seq_len, pred_len):
    self.data = data
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.target_slice = slice(0, None)

    self._read_data()

  def _read_data(self):

    n = len(self.data)

    train_df = self.data[:int(n * 0.8)]
    val_df = self.data[int(n * 0.8): int(n * 0.9)]
    test_df = self.data[int(n * 0.9):]

    self.scaler = StandardScaler()
    self.scaler.fit(train_df.values)

    def scale_df(df, scaler):
      data = scaler.transform(df.values)
      return pd.DataFrame(data, index=df.index, columns=df.columns)

    self.train_df = scale_df(train_df, self.scaler)
    self.val_df = scale_df(val_df, self.scaler)
    self.test_df = scale_df(test_df, self.scaler)
    self.n_feature = self.train_df.shape[-1]

  def _split_window(self, data):
    inputs = data[:, : self.seq_len, :]
    labels = data[:, self.seq_len :, self.target_slice]
    inputs.set_shape([None, self.seq_len, None])
    labels.set_shape([None, self.pred_len, None])
    return inputs, labels

  def _make_dataset(self, data, shuffle=True):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(data=data, targets=None, sequence_length=(self.seq_len + self.pred_len), sequence_stride=1, shuffle=shuffle, batch_size=self.batch_size)
    ds = ds.map(self._split_window)
    return ds

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

  def get_train(self, shuffle=True):
    return self._make_dataset(self.train_df, shuffle=shuffle)

  def get_val(self):
    return self._make_dataset(self.val_df, shuffle=False)

  def get_test(self):
    return self._make_dataset(self.test_df, shuffle=False)