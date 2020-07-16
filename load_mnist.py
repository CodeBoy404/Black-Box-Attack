import os
import struct
import numpy as np
import tensorflow as tf


def load_mnist(path, kind='train'):  # 设置kind的原因：方便我们之后打开测试集数据，扩展程序
    """Load MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        # 'I': 一个无符号整数，大小为4个字节
        # '>II': 读取两个无符号整数，即8个字节
        # 将文件中指针定位到数据集开头处，file.read(8)就是把文件的读取指针放到第九个字节开头处
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        # print(magic, n)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)
        # print(magic, num, rows, cols)

    return images, labels


# if __name__ == '__main__':
#     x_train, y_train = load_mnist('mnist_data', kind='train')
#     x_test, y_test = load_mnist('mnist_data', kind='test')
#
#     x_train = x_train[:, :, :, np.newaxis]
#     x_test = x_test[:, :, :, np.newaxis]
#
#     n_classes = 10
#     y_train = tf.keras.utils.to_categorical(y_train, n_classes)
#     y_test = tf.keras.utils.to_categorical(y_test, n_classes)
#
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255
