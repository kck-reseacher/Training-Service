import tensorflow as tf

class Model(tf.keras.Model):
  """Fully linear model."""

  def __init__(self, n_channel, pred_len):
    super().__init__()
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(pred_len * n_channel)
    self.reshape = tf.keras.layers.Reshape((pred_len, n_channel))

  def call(self, x):
    # x: [Batch, Input length, Channel]
    x = self.flatten(x)
    x = self.dense(x)
    x = self.reshape(x)
    return x  # [Batch, Output length, Channel]