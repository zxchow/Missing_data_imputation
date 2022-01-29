import tensorflow as tf
from MRNN import MyLinear
import numpy as np

"""
Implementation of paper: "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
"""


class GRU_D_layer(tf.keras.layers.Layer):
    def __init__(self, input_dimensions, hidden_size):
        super(GRU_D_layer, self).__init__()
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        self.dense_decay = MyLinear(input_dimensions, hidden_size)
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(
            tf.random.truncated_normal(shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            trainable=True)
        self.Wz = tf.Variable(
            tf.random.truncated_normal(shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            trainable=True)
        self.Wh = tf.Variable(
            tf.random.truncated_normal(shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01),
            trainable=True)

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(
            tf.random.truncated_normal(shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            trainable=True)
        self.Uz = tf.Variable(
            tf.random.truncated_normal(shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            trainable=True)
        self.Uh = tf.Variable(
            tf.random.truncated_normal(shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01),
            trainable=True)

        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.random.truncated_normal(shape=(self.hidden_size,), mean=0, stddev=0.01))
        self.bz = tf.Variable(tf.random.truncated_normal(shape=(self.hidden_size,), mean=0, stddev=0.01))
        self.bh = tf.Variable(tf.random.truncated_normal(shape=(self.hidden_size,), mean=0, stddev=0.01))

    def call(self, x, h_decay, init_ht=None):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        hidden_seq = []
        if init_ht is None:
            h_tm1 = tf.zeros([batch_size, self.hidden_size])
        else:
            h_tm1 = init_ht
        for t in range(sequence_length):
            x_t = x[:, t, :]
            # h_tm1 = tf.exp(-tf.nn.relu(self.dense_decay(x_t))) * h_tm1
            h_tm1 = h_decay[:, t:t+1] * h_tm1
            z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
            r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

            # Definition of h~_t
            h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
            hidden_seq.append(h_t)
        hidden_seq = tf.stack(hidden_seq, axis=1)
        return hidden_seq

    def __call__(self, inputs, h_decay):
        return self.call(inputs, h_decay)


class GRU_D(tf.keras.Model):
    def __init__(self, input_size, rnn_units, sequence_length):
        super(GRU_D, self).__init__()
        self.gru = GRU_D_layer(input_size, rnn_units)
        self.decay1 = MyLinear(sequence_length, sequence_length)
        self.decay2 = MyLinear(sequence_length * input_size, sequence_length)
        self.dense_output = MyLinear(rnn_units, input_size)
        self.learning_rate = 1
        self.c = 0
        self.model_name = 'GRU_D'

    def time_interval(self, mask):
        # mask shape: batch * channel * sequence
        mask = np.array(mask)
        sequence_length = mask.shape[2]
        interval = np.ones_like(mask)
        for i in range(1, sequence_length):
            interval[:, :, i] = interval[:, :, i] + np.multiply(interval[:, :, i - 1], 1 - mask[:, :, i - 1])
        interval[:, :, 0] = 0
        return tf.convert_to_tensor(interval)

    def last_exist(self, x, mask):
        sequence_length = x.shape[2]
        x_exist = tf.identity(x)
        x_exist = np.array(x_exist)
        for i in range(1, sequence_length):
            x_exist[:, :, i] = mask[:, :, i] * x_exist[:, :, i] + (1 - mask[:, :, i]) * x_exist[:, :, i - 1]
        return tf.convert_to_tensor(x_exist)


    def call(self, x):
        # x_decay = x[:, :, :, 1:2]  # time information
        channel_size = x.shape[1]
        sequence_length = x.shape[2]
        raw_data = x[:, :, :, 0]
        mask = x[:, :, :, 2]
        x_exist = self.last_exist(raw_data, mask)
        x_decay = self.time_interval(mask)
        x_decay1 = tf.exp(-tf.nn.relu(self.decay1(x_decay)))
        x_decay2 = tf.exp(-tf.nn.relu(self.decay2(tf.reshape(x_decay, [x.shape[0], sequence_length * channel_size]))))
        x_new = mask * x[:, :, :, 0] + (1 - mask) * (
                x_decay1 * x_exist + (1 - x_decay1) * tf.math.reduce_mean(x_exist, axis=1, keepdims=True))

        # x = tf.concat([x_new, x[:, :, :, 1:]], axis=3)
        x_new = tf.transpose(x_new, [0, 2, 1])
        # x_new = x_new[:, :, :, 0]
        # x = tf.reshape(x, [-1, x.shape[-2], x.shape[-1]])
        x_new = self.gru(x_new, x_decay2)
        x_new = self.dense_output(x_new)
        # x = tf.reshape(x, [batch_size, -1, x.shape[-2], x.shape[-1]])
        x_new = tf.expand_dims(x_new, axis=-1)
        x_new = tf.transpose(x_new, [0, 2, 1, 3])
        # x = tf.concat([x_new, x[:, :, :, 1:]], axis=-1)
        return x_new

    def __call__(self, inputs):
        return self.call(inputs)


if __name__ == '__main__':
    a = tf.random.uniform([16, 5, 20, 3])
    my_model = GRU_D(a.shape[1], 32, a.shape[2])
    b = my_model(a)
    print(b.shape)
