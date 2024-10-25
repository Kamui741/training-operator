import os
import sys
import time
import argparse
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
    with np.load(FLAGS.data_dir) as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    # 将数据展平并归一化
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    return (x_train, y_train), (x_test, y_test)

def create_datasets(x_train, y_train, x_test, y_test):
    """创建训练和测试数据集"""
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(FLAGS.batch_size).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    return train_dataset, test_dataset

def build_model():
    """构建模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(1 - FLAGS.dropout),
        tf.keras.layers.Dense(10)
    ])
    return model

def main(_):
    # 检查是否有可用 GPU
    check_gpu()

    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_data()

    # 创建数据集
    train_dataset, test_dataset = create_datasets(x_train, y_train, x_test, y_test)

    # 构建模型
    model = build_model()

    # 定义损失函数和优化器
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    # 编译模型
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # 记录训练开始时间
    start_time = time.time()

    # 训练模型
    model.fit(train_dataset, epochs=FLAGS.max_steps // (len(x_train) // FLAGS.batch_size),
              validation_data=test_dataset,
              steps_per_epoch=len(x_train) // FLAGS.batch_size)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Accuracy: {test_acc:.4f}')

    # 保存模型
    model.save(FLAGS.model_dir)

    # 计算并打印训练时长
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow MNIST Training')
    parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/data/mnist/mnist.npz', help='Path to the MNIST data file')
    parser.add_argument('--log_dir', type=str, default='/data/logs', help='Summaries log directory')
    parser.add_argument('--model_dir', type=str, default='/data/model/model.ckpt', help='Path to save the trained model')

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
