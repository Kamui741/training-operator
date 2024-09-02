'''
Author: ChZheng
Date: 2024-08-29 19:27:09
LastEditTime: 2024-08-29 19:27:12
LastEditors: ChZheng
Description:
FilePath: /笔记/Users/apple/go/src/github.com/training-operator/examples/sdk/tftrain.py
'''
import os
import tensorflow as tf
import horovod.tensorflow as hvd


def main(args):
    # 初始化 Horovod
    hvd.init()

    # Pin GPU to be used to process local rank
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # 加载数据集，并使用自定义路径
    mnist_data = np.load(args.data_path)
    mnist_images = mnist_data['x_train']
    mnist_labels = mnist_data['y_train']

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64))
    )
    dataset = dataset.repeat().shuffle(10000).batch(128)

    # 构建模型
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss = tf.losses.SparseCategoricalCrossentropy()

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(0.001 * hvd.size())

    # 自定义模型保存路径
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = mnist_model(images, training=True)
            loss_value = loss(labels, probs)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        if first_batch:
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value

    # Horovod: adjust number of steps based on number of GPUs.
    for batch, (images, labels) in enumerate(dataset.take(10000 // hvd.size())):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 10 == 0 and hvd.rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)


if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Horovod TensorFlow MNIST Training')

    # 添加完整数据路径、模型目录和日志目录参数
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset file including filename')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the log directory')

    args = parser.parse_args()

    main(args)
