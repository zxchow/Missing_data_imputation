import tensorflow as tf
import numpy as np
import statistics

"""
Implementation of paper 'Missing Value Imputation on Multidimensional Time Series'
"""


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttentionMVI(tf.keras.layers.Layer):
    def __init__(self, d_model=512, num_heads=8, kernel_size=3, window_size=10, l=5):
        super(MultiHeadAttentionMVI, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.L = l
        self.learning_rate = 0.0001
        self.c = 10
        self.model_name = 'MVI'

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(self.window_size)

    def split_heads(self, x, batch_size, channel_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, channel_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def mean_module(self, data, mask):
        data *= mask
        return tf.reduce_mean(data, axis=-1, keepdims=True)[:, :, 1:-1]

    def kernel_module(self, data, mask, l):
        batch_size = data.shape[0]
        channel_size = data.shape[1]
        seq_length = data.shape[2]
        data = data * mask
        new_data = np.array(tf.identity(data))
        new_var = np.array(tf.ones_like(data))
        new_w = np.array(tf.ones_like(data))

        for b in range(batch_size):
            for c in range(channel_size):
                for s in range(self.window_size, seq_length - self.window_size, 1):
                    if data[b, c, s] != 0:
                        continue
                    data_clip = tf.unstack(data[b], axis=0)
                    del data_clip[c]
                    data_clip = tf.stack(data_clip, axis=0)
                    index_list = []
                    data_list = []
                    for i in range(-l, l + 1):
                        if i == 0:
                            continue
                        if all(tf.math.equal(data_clip[:, s + i], data_clip[:, s])):
                            data_list.append(float(data[b, c, s + i]))
                            index_list.append(1 / abs(i))
                    if len(data_list) == 0:
                        continue
                    else:
                        new_data[b, c, s] = sum(np.array(data_list) * np.array(index_list)) / sum(index_list)
                        new_var[b, c, s] = statistics.variance(data_list)
                        new_w[b, c, s] = sum(index_list)
        new_w = tf.convert_to_tensor(new_w)
        new_var = tf.convert_to_tensor(new_var)
        new_data = tf.convert_to_tensor(new_data)
        res = tf.concat([new_data, new_var, new_w], axis=-1)
        return tf.reshape(res, [batch_size, channel_size, -1, 3 * self.window_size])[:, :, 1:-1]

    def call(self, x):
        time = x[:, :, :, 1]
        mask = x[:, :, :, 2]
        x = x[:, :, :, 0]

        batch_size, channel_size = tf.shape(x)[0], tf.shape(x)[1]

        # kernel_feature = self.kernel_module(x, mask, self.L // 2)

        x = tf.reshape(x, [batch_size, channel_size, -1, self.window_size])
        mask = tf.reshape(mask, [batch_size, channel_size, -1, self.window_size])
        mean_feature = self.mean_module(x, mask)
        mask = mask[:, :, 1:-1]
        mask = tf.reduce_min(mask, axis=-1)
        # mask = create_padding_mask(mask)
        look_ahead_mask = create_look_ahead_mask(mask.shape[-1])
        padding_mask = create_padding_mask_MVI(mask)
        mask = tf.maximum(look_ahead_mask, padding_mask)
        pos_encoding = positional_encoding(x.shape[2], x.shape[-1])
        pos_encoding = tf.concat([pos_encoding for _ in range(batch_size * channel_size)], axis=0)
        pos_encoding = tf.reshape(pos_encoding, [batch_size, channel_size, x.shape[2], x.shape[-1]])
        x += pos_encoding
        q = self.wq(tf.concat([x[:, :, 2:], x[:, :, :-2]], axis=-1))  # (batch_size, channel, seq_len, d_model)
        k = self.wk(tf.concat([x[:, :, 2:], x[:, :, :-2]], axis=-1))  # (batch_size, channel, seq_len, d_model)
        v = self.wv(x[:, :, 1:-1])  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, channel_size)  # (batch_size, channel, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, channel_size)  # (batch_size, channel, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, channel_size)  # (batch_size, channel, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 1, 3, 2, 4])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, channel_size, -1, self.d_model))
        # (batch_size, channel, seq_len_q, d_model)

        # concat_feature = tf.concat([concat_attention, mean_feature, kernel_feature], axis=-1)
        concat_feature = tf.concat([concat_attention, mean_feature], axis=-1)

        output = self.dense(concat_feature)  # (batch_size, channel, seq_len, window)

        return tf.expand_dims(tf.reshape(output, [output.shape[0], output.shape[1], -1]), axis=-1)

    def __call__(self, x):
        return self.call(x)


def create_padding_mask_MVI(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, :, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


if __name__ == '__main__':
    dummy_input = tf.random.normal([16, 5, 100, 3])
    raw_data = dummy_input[:, :, :, 0]
    # print(positional_encoding(200, 3))
    data_time = dummy_input[:, :, :, 1]
    # mask = dummy_input[:, :, :, 2]
    mask = tf.concat([tf.ones([16, 5, 50]), tf.zeros([16, 5, 50])], axis=-1)
    x = tf.stack([raw_data, data_time, mask], axis=-1)
    # mask 1 represent true value, 0 represent missing value

    model = MultiHeadAttentionMVI()
    b = model(x)
    print(b.shape)
