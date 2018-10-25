import numpy as np
import tensorflow as tf
import os


class DataReader:

    def __init__(self, folder_name):
        self.folder_name = folder_name

    def load_images_train(self, indices_mb):
        return self.load_npy_files(["image_train%02d.npy" % idx for idx in indices_mb])

    def load_images_test(self, indices_mb):
        return self.load_npy_files(["image_test%02d.npy" % idx for idx in indices_mb])

    def load_labels_train(self, indices_mb):
        return self.load_npy_files(["label_train%02d.npy" % idx for idx in indices_mb])

    def load_npy_files(self, file_names):
        images = [np.float32(np.load(os.path.join(self.folder_name, fn))) for fn in file_names]
        return np.expand_dims(np.stack(images, axis=0), axis=4)


def resize_volume(image, size, method=0, name='resize_volume'):
    # size is [depth, height width]
    # image is Tensor with shape [batch, depth, height, width, channels]
    shape = image.get_shape().as_list()
    with tf.variable_scope(name):
        reshaped2d = tf.reshape(image, [-1, shape[2], shape[3], shape[4]])
        resized2d = tf.image.resize_images(reshaped2d, [size[1], size[2]], method)
        reshaped2d = tf.reshape(resized2d, [shape[0], shape[1], size[1], size[2], shape[4]])
        permuted = tf.transpose(reshaped2d, [0, 3, 2, 1, 4])
        reshaped2db = tf.reshape(permuted, [-1, size[1], shape[1], shape[4]])
        resized2db = tf.image.resize_images(reshaped2db, [size[1], size[0]], method)
        reshaped2db = tf.reshape(resized2db, [shape[0], size[2], size[1], size[0], shape[4]])
        return tf.transpose(reshaped2db, [0, 3, 2, 1, 4])
