import sys
sys.path.append('../')
import os
import numpy as np
import tensorflow as tf
from load_mnist import load_mnist

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    x_train, y_train = load_mnist('../mnist_data', kind='train')
    x_test, y_test = load_mnist('../mnist_data', kind='test')

    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    n_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    inputs = tf.keras.Input(shape=(28, 28, 1), name='data')
    x = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='relu')(x)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='lenet')

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    # 保存
    checkpoint_path = "./checkpoints/"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=1)

    history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test),
                        callbacks=[cp_callback])
