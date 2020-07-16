# -*- coding: utf-8 -*-
# @Time    : 2020/7/16 11:01
# @Author  : Antin
# @FileName: attack_antin.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append('../')
import os
import tensorflow as tf
from load_mnist import load_mnist
import numpy as np
import random
import matplotlib.pyplot as plt     # plt用于显示图片


def image_show(image):
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def normalization(image):
    image = image.reshape(-1, 28, 28, 1).astype('float32')
    image /= 255
    return image


def is_adversarial(model, original_label, generate):
    generate_label = np.argmax(model.predict(normalization(random_)))
    print(generate_label)
    if generate_label == original_label:
        return False
    else:
        return True


steps = 1
spherical_step = 1e-2
source_step = 1e-2


if __name__ == '__main__':
    # 先读取数据
    x_train, y_train = load_mnist('mnist_data', kind='train')
    x_test, y_test = load_mnist('mnist_data', kind='test')
    images = np.append(x_train, x_test).reshape(-1, 28, 28)
    image = random.choice(images)

    image_show(image)

    input_ = image
    image = normalization(image)

    checkpoint_path = "net1/checkpoints/saved_model.pb"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = tf.keras.models.load_model(checkpoint_dir)
    model.compile(loss='categorical_crossentropy', metrics=['CategoricalAccuracy'])

    # 原始图片的标签,这里应该不算查询次数
    original_label = np.argmax(model.predict(image))
    print(original_label)

    # 1.先生成对抗样本
    random_ = np.trunc(np.random.uniform(low=0, high=255, size=input_.shape))

    image_show(random_)

    while not is_adversarial(model, original_label, random_):
        random_ = np.trunc(np.random.uniform(low=0, high=255, size=input_.shape))
        generate_label = np.argmax(model.predict(normalization(random_)))
    # 最后出来的random_一定是对抗样本, random是第k步，random_是第k-1步
    # 2. 循环

    def is_flag1(disturbance, random_):
        print(random_ + disturbance)
        if (random_ + disturbance).any() > 255:
            return False
        else:
            return True

    for i in range(1, steps+1):
        # 3. 找到符合条件的扰动
        # 3.1 从二维高斯分布中选取扰动
        disturbance = np.random.uniform(low=0, high=255, size=input_.shape)
        # random = np.trunc(np.random.uniform(low=0, high=255, size=input_.shape))
        # if is_adversarial(model, original_label, random+random_):
        #     random_ = random_ + random
        # else:
        #     random_ = random_
        flag1, flag2, flag3 = True, True, True

        flag1 = is_flag1(disturbance, random_)

        while not flag1:
            pass

