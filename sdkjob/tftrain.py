'''
Author: ChZheng
Date: 2024-10-17 14:29:53
LastEditTime: 2024-10-17 14:30:58
LastEditors: ChZheng
Description:
FilePath: /horovod/Users/apple/go/src/github.com/training-operator/sdkjob/tftrain.py
'''
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf

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
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    return (x_train, y_train), (x_test, y_test)

def train():
    start_time = time.time()  # 记录训练开始时间

    # 检查是否有可用 GPU
    check_gpu()

    # 导入数据
    (x_train, y_train), (x_test, y_test) = load_data()

    # 构建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(FLAGS.batch_size).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    # 构建模型（使用 Keras Sequential API）
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='relu', input_shape=(784,)),  # 输入层
        tf.keras.layers.Dropout(1 - FLAGS.dropout),  # dropout
        tf.keras.layers.Dense(10)  # 输出层
    ])

    # 定义损失函数和优化器
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    # 编译模型
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_dataset, epochs=FLAGS.max_steps // len(x_train), validation_data=test_dataset)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Accuracy: {test_acc}')

    # 保存模型
    model.save(FLAGS.model_path)

    end_time = time.time()  # 记录训练结束时间
    duration = end_time - start_time
    print(f"Training duration: {duration} seconds")

def main(_):
    if tf.io.gfile.exists(FLAGS.log_dir):
        tf.io.gfile.rmtree(FLAGS.log_dir)  # 删除目录
    tf.io.gfile.makedirs(FLAGS.log_dir)    # 创建目录
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
    parser.add_argument('--data_path', type=str, default='/data/mnist/mnist.npz', help='Path to the MNIST data file')
    parser.add_argument('--log_dir', type=str, default='/data/logs', help='Summaries log directory')
    parser.add_argument('--model_path', type=str, default='/data/model/model.ckpt', help='Path to save the trained model')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
