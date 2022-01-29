import tensorflow as tf

"""
Implementation of paper:
"Estimating Missing Data in Temporal Data Streams Using Multi-directional Recurrent Neural Networks"
"""


class MyLinear(tf.keras.layers.Layer):
    def __init__(self, input_dim, units):
        super(MyLinear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

    def __call__(self, inputs):
        return self.call(inputs)


class MRNN(tf.keras.Model):

    def __init__(self, channel_size, rnn_units=1):
        super(MRNN, self).__init__()
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn_f = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, return_state=False, activation='relu')
        self.rnn_b = tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True, return_state=False, activation='relu',
                                               go_backwards=True)
        self.dense1 = MyLinear(rnn_units * 2, 1)
        self.dense2 = MyLinear(channel_size, channel_size)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.learning_rate = 0.001
        self.c = 1
        self.model_name = 'MRNN'

    def call(self, inputs, training=True):
        batch_size = inputs.shape[0]
        x = tf.reshape(inputs, [-1, inputs.shape[-2], inputs.shape[-1]])
        # x = tf.transpose(inputs, [0, 2, 1, 3])
        x_f = x[:, :-2]
        x_b = x[:, 2:]
        # states = self.rnn_b.get_initial_state(x)
        x_f = self.rnn_f(x_f, training=training)
        x_b = self.rnn_b(x_b, training=training)
        x = tf.concat([x_f, x_b], axis=-1)
        # x: (batch * channel) * seq  * 2units
        x = self.dense1(x)
        x = tf.reshape(x, [batch_size, -1, x.shape[-2], x.shape[-1]])
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        return x

    def __call__(self, inputs):
        return self.call(inputs)


if __name__ == '__main__':
    # input batch * channel * seq_length * feature_size
    a = tf.random.uniform([16, 5, 20, 3])
    my_model = MRNN(a.shape[1], 32)
    b = my_model(a)
    print(b.shape)
    # output: batch * channel * sequence * 1
