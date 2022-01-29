import tensorflow as tf
from MRNN import MyLinear


"""
Implementation of paper "SSIM—A Deep Learning Approach for Recovering Missing Time Series Sensor Data"
"""
class SSIM(tf.keras.Model):
    def __init__(self, encoder_units, channel_size=5, feature_size=3):
        super(SSIM, self).__init__()
        self.data_dimension = channel_size
        self.rnn_units = encoder_units * 2
        self.encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(encoder_units, return_sequences=True))
        self.decoder = tf.keras.layers.LSTM(self.rnn_units)
        self.attention = MyLinear(encoder_units*2 + self.data_dimension, 1)
        self.dense = MyLinear(encoder_units*2 + self.rnn_units, self.data_dimension)
        self.learning_rate = 0.1
        self.c = 1
        self.model_name = 'SSIM'

    def call(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        batch_size, sequence_length, channel_size, feature_size = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        masking = x[:, :, 0, 2]
        x = x[:, :, :, 0]
        # x = tf.reshape(x, [batch_size, sequence_length, -1])
        # masking = tf.reduce_any(x != 0, axis=2)
        encoder_output = self.encoder(x, mask=tf.cast(masking, tf.bool))

        seq_res = []

        hidden = encoder_output[:, -1, :]
        x_t = x[:, 0, :]
        for t in range(1, sequence_length):
            attention_vector = tf.reduce_sum(tf.multiply(tf.nn.softmax(self.attention(
                tf.concat([tf.stack([x_t] * sequence_length, axis=1), encoder_output], axis=2)
            ), axis=1), encoder_output), axis=1)
            # hidden = self.decoder(tf.expand_dims(tf.concat([x_t, hidden, attention_vector], axis=1), axis=1))
            hidden = self.decoder(tf.expand_dims(x_t, axis=1), initial_state=[hidden, attention_vector])
            x_t = self.dense(tf.concat([hidden, attention_vector], axis=-1))
            seq_res.append(x_t)
        seq_res = tf.stack(seq_res, axis=1)
        # seq_res = tf.reshape(seq_res, [batch_size, sequence_length - 1, channel_size])
        return tf.expand_dims(tf.transpose(seq_res, [0, 2, 1]), axis=-1)

    def __call__(self, inputs):
        return self.call(inputs)



if __name__ == '__main__':
    a = tf.random.uniform([16, 5, 20, 3])
    model = SSIM(30, a.shape[1], a.shape[-1])
    b = model(a)
    print(b.shape)

