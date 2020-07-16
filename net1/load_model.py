from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('../')
import os
import tensorflow as tf
from load_mnist import load_mnist
import numpy as np
import random
import matplotlib.pyplot as plt     # plt用于显示图片
import matplotlib.image as mpimg    # 用于读取图片


if __name__ == '__main__':
    # 先读取数据
    x_train, y_train = load_mnist('../mnist_data', kind='train')
    x_test, y_test = load_mnist('../mnist_data', kind='test')

    images = np.append(x_train, x_test).reshape(-1, 28, 28)
    # image = random.choice(images)

    subset = images[np.random.choice(len(images), size=250, replace=False)]

    # plt.imshow(image)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    #
    subset = subset.reshape(-1, 28, 28, 1).astype('float32')

    np.save('250-28-28-1.npy', subset)

    subset /= 255

    checkpoint_path = "./checkpoints/saved_model.pb"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = tf.keras.models.load_model(checkpoint_dir)
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    output = model.predict(subset)
    ids = np.argmax(output, axis=1)

    np.save('250-1.npy', ids)
