import tensorflow as tf
from MRNN import MyLinear

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

    def call(self, x, init_ht=None):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        hidden_seq = []
        if init_ht is None:
            h_tm1 = tf.zeros([batch_size, self.hidden_size])
        else:
            h_tm1 = init_ht
        for t in range(sequence_length):
            x_t = x[:, t, :]
            h_tm1 = tf.exp(-tf.nn.relu(self.dense_decay(x_t))) * h_tm1
            z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
            r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

            # Definition of h~_t
            h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
            hidden_seq.append(h_t)
        hidden_seq = tf.stack(hidden_seq, axis=1)
        return hidden_seq

    def __call__(self, inputs):
        return self.call(inputs)


class GRU_D(tf.keras.Model):
    def __init__(self, input_size, rnn_units):
        super(GRU_D, self).__init__()
        self.gru = GRU_D_layer(input_size, rnn_units)
        self.decay1 = MyLinear(1, 1)
        self.decay2 = MyLinear(1, 1)
        self.dense_output = MyLinear(rnn_units, input_size)
        self.learning_rate = 0.1
        self.c = 0
        self.model_name = 'GRU_D'

    def call(self, x):
        x_decay = x[:, :, :, 1:2]  # time information
        x_decay1 = tf.exp(-tf.nn.relu(self.decay1(x_decay)))
        x_new = x[:, :, :, 2:] * x[:, :, :, 0:1] + (1 - x[:, :, :, 2:]) * (
                x_decay1 * x[:, :, :, 0:1] + (1 - x_decay1) * tf.math.reduce_mean(x[:, :, :, 0:1], axis=0,
                                                                                  keepdims=True))
        # x = tf.concat([x_new, x[:, :, :, 1:]], axis=3)
        x_new = tf.transpose(x_new, [0, 2, 1, 3])
        x_new = x_new[:, :, :, 0]
        # x = tf.reshape(x, [-1, x.shape[-2], x.shape[-1]])
        x_new = self.gru(x_new)
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
    my_model = GRU_D(a.shape[1], 32)
    b = my_model(a)
    print(b.shape)
