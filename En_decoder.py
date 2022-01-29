import tensorflow as tf
import tensorflow_addons as tfa

"""
Implementation of paper "Learning from Irregularly-Sampled Time Series: A Miss Data Perspective"
"""
def conv_ln_lrelu(out_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.ZeroPadding1D(padding=2),
            tfa.layers.SpectralNormalization(tf.keras.layers.Conv1D(out_dim, 5, 2)),
            tf.keras.layers.LeakyReLU(0.2)
        ]
    )


def dconv_bn_relu(out_dim):
    return tf.keras.Sequential([
        # tf.keras.layers.ZeroPadding1D(padding=2),
        tf.keras.layers.Conv1DTranspose(out_dim, 5, 2, padding='same', output_padding=1, use_bias=False),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.ReLU()])


class Encoder(tf.keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.cconv = tf.keras.layers.Conv1D(hidden_size, 5)
        self.ls = tf.keras.Sequential([
            tf.keras.layers.LeakyReLU(0.2),
            conv_ln_lrelu(128),
            conv_ln_lrelu(256),
            conv_ln_lrelu(512),
            conv_ln_lrelu(64)]
        )


    def call(self, x):
        x = tf.transpose(x, [0, 2, 1])
        x = self.cconv(x)
        x = self.ls(x)
        return x



class SeqGeneratorDiscrete(tf.keras.Model):
    def __init__(self, n_channels=5, latent_size=128):
        super().__init__()

        self.l1 = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()])

        self.l2 = tf.keras.Sequential([
            dconv_bn_relu(128),
            dconv_bn_relu(64),
            dconv_bn_relu(32),
            # tf.keras.layers.ZeroPadding1D(padding=2),
            tf.keras.layers.Conv1DTranspose(n_channels, 5, 2, padding='same', output_padding=1)])

        # self.reshape_layer = tf.keras.layers.Reshape((8, 1536))

    def call(self, z):
        h = self.l1(z)
        h = self.l2(h)
        return h


class Decoder(tf.keras.Model):
    def __init__(self, grid_decoder):
        super().__init__()
        self.grid_decoder = grid_decoder

    def call(self, code, time, mask):
        """
        Args:
            code: shape (batch_size, latent_size)
            time: shape (batch_size, channels, max_seq_len)
            mask: shape (batch_size, channels, max_seq_len)

        Returns:
            interpolated tensor of shape (batch_size, max_seq_len)
        """
        # shape of x: (batch_size, n_channels, dec_ref)
        x = self.grid_decoder(code)
        x = tf.transpose(x, [0, 2, 1])
        return x


class PVAE(tf.keras.Model):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = 0.0001
        self.c = 2
        self.model_name = 'PVAE'

    def call(self, inputs):
        data = inputs[:, :, :, 0]
        time = inputs[:, :, :, 1]
        mask = inputs[:, :, :, 2]
        z = self.encoder(data * mask)
        x_recon = self.decoder(z, time, mask)
        return tf.expand_dims(x_recon, axis=-1)

    def __call__(self, inputs):
        return self.call(inputs)


def generate_model(hidden_size=32, channel_size=5):
    encoder = Encoder(hidden_size)
    grid_decoder = SeqGeneratorDiscrete(channel_size)
    decoder = Decoder(grid_decoder=grid_decoder)
    my_model = PVAE(encoder, decoder)
    return my_model


if __name__ == '__main__':
    encoder = Encoder(64)
    grid_decoder = SeqGeneratorDiscrete(5)
    decoder = Decoder(grid_decoder=grid_decoder)
    my_model = PVAE(encoder, decoder)
    dummy_input = tf.random.normal([64, 5, 100, 3])
    x_recon = my_model(dummy_input)
    print(x_recon.shape)
