import tensorflow as tf

"""
Implementation of paper "NAOMI: Non-Autoregressive Multiresolution Sequence Imputation"
"""

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total



class NAOMI(tf.keras.Model):

    def __init__(self, batch_size, y_dim):
        super(NAOMI, self).__init__()

        self.stochastic = False
        self.y_dim = y_dim
        self.rnn_dim = 100
        self.dims = {}
        self.n_layers = 1
        self.networks = {}
        self.highest = 8
        self.batch_size = batch_size

        self.gru = tf.keras.layers.GRU(self.rnn_dim, time_major=True)
        self.back_gru = tf.keras.layers.GRU(self.rnn_dim, time_major=True)
        self.learning_rate = 0.01
        self.c = 0
        self.model_name = 'NAOMI'

        step = 1
        while step <= self.highest:
            l = str(step)
            self.dims[l] = 50
            dim = self.dims[l]

            curr_level = {}
            curr_level['dec'] = tf.keras.Sequential(
                tf.keras.layers.Dense(dim, input_shape=(2*self.rnn_dim,), activation='relu'))
            curr_level['mean'] = tf.keras.layers.Dense(self.y_dim, input_shape=(dim,))

            # curr_level = nn.ModuleDict(curr_level)

            self.networks[l] = curr_level

            step = step * 2

        # self.networks = nn.ModuleDict(self.networks)

    def data_preparation(self, a):
        raw_data = a[:, :, :, 0]
        mask = tf.transpose(a[:, :1, :, 2], [2, 0, 1])
        mask = tf.reshape(mask, [mask.shape[0], mask.shape[1], -1])
        raw_data = tf.transpose(raw_data, [2, 0, 1])
        inputs = tf.concat([raw_data, mask], axis=-1)
        data = []
        # inputs = tf.concat((tf.ones((5, 5, 4)), tf.zeros((10, 5, 4)), tf.ones((5, 5, 4))), axis=0)
        for j in range(inputs.shape[0]):
            data.append(inputs[j:j + 1])
        return data

    def call(self, data_list):
        data_list = self.data_preparation(data_list)
        # data_list: seq_length * (1 * batch * 11)
        seq_len = len(data_list)
        h = tf.Variable(tf.zeros((self.batch_size, self.rnn_dim)))

        h_back_dict = {}
        h_back = tf.Variable(tf.zeros((self.batch_size, self.rnn_dim)))
        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t + 1] = h_back
            state_t = data_list[t]
            h_back = self.back_gru(state_t, h_back)

        curr_p = 0
        h = self.gru(data_list[curr_p][:, :, 1:], h)
        while curr_p < seq_len - 1:
            if data_list[curr_p + 1][0, 0, -1] == 1:
                curr_p += 1
                h = self.gru(data_list[curr_p][:, :, 1:], h)
            else:
                next_p = curr_p + 1
                while next_p < seq_len and data_list[next_p][0, 0, -1] == 0:
                    next_p += 1

                step_size = 1
                while curr_p + 2 * step_size <= next_p and step_size <= self.highest:
                    step_size *= 2
                if step_size > 1:
                    step_size = step_size // 2

                self.interpolate(data_list, curr_p, h, h_back_dict, step_size)

        return tf.expand_dims(tf.transpose(tf.concat(data_list, axis=0)[:, :, 1:], [1, 2, 0]), axis=-1)

    def interpolate(self, data_list, curr_p, h, h_back_dict, step_size):
        # print("interpolating:", len(ret), step_size)
        h_back = h_back_dict[curr_p + 2 * step_size]
        curr_level = self.networks[str(step_size)]

        dec_t = curr_level['dec'](tf.concat([h, h_back], 1))
        dec_mean_t = curr_level['mean'](dec_t)

        state_t = dec_mean_t

        added_state = tf.expand_dims(state_t, axis=0)
        has_value = tf.Variable(tf.ones((added_state.shape[0], added_state.shape[1], 1)))
        added_state = tf.concat([added_state, has_value], 2)

        if step_size > 1:
            right = curr_p + step_size
            left = curr_p + step_size // 2
            h_back = h_back_dict[right + 1]
            h_back = self.back_gru(added_state, h_back)
            h_back_dict[right] = h_back

            zeros = tf.Variable(tf.zeros((added_state.shape[0], added_state.shape[1], self.y_dim + 1)))
            for i in range(right - 1, left - 1, -1):
                h_back = self.back_gru(zeros, h_back)
                h_back_dict[i] = h_back

        data_list[curr_p + step_size] = added_state

    def __call__(self, inputs):
        return self.call(inputs)



if __name__ == '__main__':
    a = tf.ones([64, 5, 100, 3])
    # raw_data = a[:, :, :, 0]
    # mask = tf.transpose(a[:, :1, :, 2], [2, 0, 1])
    # mask = tf.reshape(mask, [mask.shape[0], mask.shape[1], -1])
    # raw_data = tf.transpose(raw_data, [2, 0, 1])
    # inputs = tf.concat([raw_data, mask], axis=-1)
    model = NAOMI(64, 5)
    # data = []
    # # inputs = tf.concat((tf.ones((5, 5, 4)), tf.zeros((10, 5, 4)), tf.ones((5, 5, 4))), axis=0)
    # for j in range(inputs.shape[0]):
    #     data.append(inputs[j:j+1])
    # # 20, 16, 5
    res = model(a)
    print(res.shape)
