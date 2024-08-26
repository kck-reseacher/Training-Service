import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Dense, Dropout

def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""

    norm = (LayerNormalization if norm_type == 'L' else BatchNormalization)

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear
    x = norm(axis=[-2, -1])(res)
    x = Dense(ff_dim, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = Dropout(dropout)(x)
    return x + res


def build_model(input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice):
    """Build TSMixer model."""

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    for _ in range(n_block):
      x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice:
      x = x[:, :, target_slice]

    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

    return tf.keras.Model(inputs, outputs)