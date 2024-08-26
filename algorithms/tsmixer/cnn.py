import tensorflow as tf

class Model(tf.keras.Model):
  """CNN model."""

  def __init__(self, n_channel, pred_len, kernel_size):
    super().__init__()
    self.cnn = tf.keras.layers.Conv1D(
        n_channel, kernel_size, padding='same', input_shape=(None, n_channel)
    )
    self.dense = tf.keras.layers.Dense(pred_len)

  def call(self, x):
    x = self.cnn(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = self.dense(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    return x