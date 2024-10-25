import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

FLAGS = None

def check_gpu():
    """自动检测 GPU 可用性"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
    else:
        print("No GPUs available, using CPU.")

def load_data():
    """从本地加载 MNIST 数据集"""
    with np.load(FLAGS.data_path) as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    # 将数据展平
    x_train = x_train[..., tf.newaxis] / 255.0  # 归一化并调整维度
    x_test = x_test[..., tf.newaxis] / 255.0

    return (x_train, y_train), (x_test, y_test)

def main(_):
    # 初始化 Horovod
    hvd.init()

    # Pin GPU to be used to process local rank
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # 检查是否有可用 GPU
    check_gpu()

    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_data()

    # 创建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.repeat().shuffle(10000).batch(128)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(128)

    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 定义损失函数和优化器
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Horovod: adjust learning rate based on number of GPUs.
    optimizer = tf.keras.optimizers.Adam(0.001 * hvd.size())

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    checkpoint_dir = os.path.join(FLAGS.model_dir, 'checkpoints')
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss_fn(labels, probs)

        # Horovod: 使用 Horovod 分布式 GradientTape
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return loss_value

    # Horovod: 根据 GPU 数量调整 step 数
    start_time = time.time()
    for batch, (images, labels) in enumerate(train_dataset.take(10000 // hvd.size())):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 10 == 0 and hvd.rank() == 0:
            print(f'Step #{batch}\tLoss: {loss_value:.6f}')

    # 计算训练时长
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds")

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Horovod TensorFlow MNIST Training')

    # 添加完整数据路径、模型目录和日志目录参数
    parser.add_argument('--data_path', type=str, default='/data/mnist/mnist.npz', help='Path to the dataset file')
    parser.add_argument('--model_dir', type=str, default='/data/model', help='Path to the model directory')
    parser.add_argument('--log_dir', type=str, default='/data/logs', help='Path to the log directory')

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
