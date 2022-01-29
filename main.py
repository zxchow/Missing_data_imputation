import numpy as np
import pandas as pd
import SSIM
import DeepMVI
import En_decoder
import NAOMI
import tensorflow as tf
import time
import MRNN
import GRU_D


def load_data(file_path='./HSIall.csv'):
    data_length = 100
    raw_data = pd.read_csv(file_path, encoding='gbk', header=None).iloc[1:, 2:7].to_numpy(dtype=np.float32)
    # raw_data = tf.convert_to_tensor(raw_data)
    raw_data = raw_data.T
    raw_data_length = raw_data.shape[1]
    raw_data = raw_data[:, :raw_data_length // data_length * data_length].reshape(5, -1, data_length)
    batch_size = raw_data.shape[1]
    time = np.arange(data_length)
    time = np.stack([time for _ in range(5 * raw_data.shape[1])]).reshape(5, -1, data_length)
    # print(time.shape)
    mask = np.concatenate((np.ones([5, batch_size, int(data_length * 0.4)]),
                           np.zeros([5, batch_size, int(data_length * 0.2)]),
                           np.ones([5, batch_size, int(data_length * 0.4)])), axis=-1)
    # mask_data = raw_data * mask
    combined = np.stack((raw_data, time, mask), axis=-1)
    combined = np.transpose(combined, [1, 0, 2, 3])
    # print(combined.shape)
    # print(raw_data.shape)
    # 967 * 5 * 100 * 3
    return combined


def loss_function(raw_data, imputed_data, c=0, contain_loss=False):
    if contain_loss:
        return imputed_data[1]
    else:
        data_length = raw_data.shape[2]
        raw_data = raw_data[:, :, int(data_length * 0.4):int(data_length * 0.6), 0]
        imputed_data = imputed_data[:, :, int(data_length * 0.4) - c:int(data_length * 0.6) - c, 0]
        return tf.keras.losses.mean_squared_error(raw_data, imputed_data)


def model_run(model, epoch_count, raw_data):
    batch_size = 64
    channel_size = 5
    feature_size = 3
    raw_data = tf.cast(raw_data, tf.float32)
    # raw_data = np.ones((967, 5, 100, 3))
    dataset_size = raw_data.shape[0]
    contain_loss = False
    raw_data = tf.data.Dataset.from_tensor_slices(raw_data)
    raw_data = raw_data.shuffle(dataset_size, reshuffle_each_iteration=False)
    train_size = int(0.7 * dataset_size) // batch_size * batch_size
    test_size = (dataset_size - train_size) // batch_size * batch_size
    train_dataset = raw_data.take(train_size).batch(batch_size)
    test_dataset = raw_data.skip(train_size).take(test_size).batch(batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)
    min_loss = float('inf')
    min_epoch = 0
    for epoch in range(epoch_count):
        loss_sum = 0
        for data in train_dataset:
            with tf.GradientTape() as tape:
                mask_data = tf.stack([data[:, :, :, 0] * data[:, :, :, 2], data[:, :, :, 1], data[:, :, :, 2]], axis=-1)
                imputed_data = model(mask_data)
                loss_value = loss_function(data, imputed_data, model.c, contain_loss)
                loss_sum += tf.reduce_sum(loss_value)
            grads = tape.gradient(loss_value, model.trainable_variables,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print("epoch: {}, loss: {}".format(epoch, (loss_sum / train_size) ** 0.5))
        test_loss_sum = 0
        for test_data in test_dataset:
            mask_test = tf.stack(
                [test_data[:, :, :, 0] * test_data[:, :, :, 2], test_data[:, :, :, 1], test_data[:, :, :, 2]], axis=-1)
            imputed_test_data = model(mask_test)
            test_loss = loss_function(test_data, imputed_test_data, model.c, contain_loss)
            test_loss_sum += tf.reduce_sum(test_loss)
        if test_loss_sum < min_loss:
            min_loss = test_loss_sum
            min_epoch = epoch
        print("epoch: {}, test loss: {}".format(epoch, (test_loss_sum / test_size) ** 0.5))
    print("minimal loss {} in epoch {}".format((min_loss / test_size) ** 0.5, min_epoch))
    return (min_loss / test_size) ** 0.5


if __name__ == '__main__':
    epoch_count = 500
    tf.random.set_seed(2022)
    np.random.seed(2022)
    channel_size = 5
    feature_size = 3
    raw_data = load_data()
    print("finish_loading")
    model0 = MRNN.MRNN(channel_size, 32)
    model1 = GRU_D.GRU_D(channel_size, 32)
    model2 = NAOMI.NAOMI(64, channel_size)
    model3 = En_decoder.generate_model(64, channel_size)
    model4 = DeepMVI.MultiHeadAttentionMVI()
    model5 = SSIM.SSIM(32, channel_size, feature_size)
    model_list = [model0, model1, model2, model3, model4, model5]
    for i, model in enumerate(model_list):
        start_time = time.time()
        min_loss = model_run(model, epoch_count, raw_data)
        end_time = time.time()
        running_time = (end_time - start_time) / epoch_count
        with open('result_{}.txt'.format(model.model_name), 'a') as f:
            f.writelines(
                "\n large miss: model{} reach a min_loss of {} with a running time of {} in large missing".format(
                    model.model_name, min_loss, running_time))
