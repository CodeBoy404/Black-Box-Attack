import sys
sys.path.append('../')
import tensorflow as tf
import os
import numpy as np
from load_mnist import load_mnist

strategy = tf.distribute.MirroredStrategy(["/cpu:0"])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

train_images, train_labels = load_mnist('../mnist_data', kind='train')
test_images, test_labels = load_mnist('../mnist_data', kind='test')

print(train_images.shape)
# 向数组添加维度 -> 新的维度 == (28, 28, 1)
# 我们这样做是因为我们模型中的第一层是卷积层
# 而且它需要一个四维的输入 (批大小, 高, 宽, 通道).
# 批大小维度稍后将添加。
train_images = train_images[..., None]
test_images = test_images[..., None]

# 获取[0,1]范围内的图像。
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

#***************************** 设置输入流水线 **********************************
BUFFER_SIZE = len(train_images)

# 设置 batch size (全局bs = 单卡bs * num_gpus)
# 单卡bs
BATCH_SIZE_PER_REPLICA = 16
# 全局bs
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 3
# 创建数据集并分发它们：
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

num_epochs = 5
learning_rate = 0.001

with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
steps_per_epoch=np.ceil(60000/GLOBAL_BATCH_SIZE)
validation_steps=np.ceil(10000/GLOBAL_BATCH_SIZE)

# 保存
checkpoint_path = "./checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=1)

model.fit(train_dataset.repeat(), epochs=num_epochs, steps_per_epoch=steps_per_epoch,
          validation_data=test_dataset.repeat(), validation_steps=validation_steps, callbacks=[cp_callback])



