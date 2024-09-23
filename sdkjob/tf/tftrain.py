from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd  # 加入 Horovod

FLAGS = None

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
    hvd.init()  # 初始化 Horovod

    # 检查是否有可用的 GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 如果存在 GPU，设置 GPU 使用策略
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Horovod 分布式 GPU 设置
        if gpus and hvd.local_rank() < len(gpus):
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    else:
        print("No GPUs detected, running on CPU")

    start_time = time.time()  # 记录训练开始时间

    # 导入数据
    (x_train, y_train), (x_test, y_test) = load_data()

    # 构建数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(FLAGS.batch_size).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(FLAGS.batch_size)

    # Horovod 分布式数据加载器处理，设置每个 worker 只处理数据的一个子集
    train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
    test_dataset = test_dataset.shard(hvd.size(), hvd.rank())

    # 创建会话并初始化变量
    sess = tf.InteractiveSession()

    # 创建多层模型

    # 输入占位符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    def weight_variable(shape):
        """创建权重变量"""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """创建偏置变量"""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """为 Tensor 附加多个摘要（用于 TensorBoard 可视化）"""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """创建简单神经网络层的可复用代码"""
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate * hvd.size())  # 调整学习率
        opt = hvd.DistributedOptimizer(opt)  # Horovod 分布式优化器
        train_step = opt.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    log_dir = FLAGS.log_dir
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()

    # Horovod 广播变量初始值，确保所有 workers 同步
    bcast = hvd.broadcast_global_variables(0)

    # 训练循环
    for i in range(FLAGS.max_steps):
        if i % 10 == 0 and hvd.rank() == 0:  # 只让 rank 0 打印
            summary, acc = sess.run([merged, accuracy], feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                if hvd.rank() == 0:  # 只让 rank 0 打印
                    print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step])
                train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()

    # 保存模型，仅在 rank 0 上保存模型
    if hvd.rank() == 0:
        saver = tf.train.Saver()
        saver.save(sess, FLAGS.model_path)

    end_time = time.time()  # 记录训练结束时间
    duration = end_time - start_time
    if hvd.rank() == 0:  # 只让 rank 0 打印
        print(f"Training duration: {duration} seconds")

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
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
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
