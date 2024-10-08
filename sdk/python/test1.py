'''
Author: ChZheng
Date: 2024-08-02 14:19:53
LastEditTime: 2024-08-02 14:37:12
LastEditors: ChZheng
Description:
FilePath: /笔记/Users/apple/go/src/github.com/training-operator/test1.py
'''
# 导入必要的模块
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train():
    # 导入数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)

    # 创建一个新的 TensorFlow 会话
    sess = tf.InteractiveSession()

    # 定义输入占位符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    # 定义神经网络结构
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            weights = weight_variable([input_dim, output_dim])
            biases = bias_variable([output_dim])
            preactivate = tf.matmul(input_tensor, weights) + biases
            activations = act(preactivate, name='activation')
            return activations

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()

    # 定义模型保存器
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # 开始训练
    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size, fake_data=FLAGS.fake_data)
        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print(f'Accuracy at step {i}: {acc}')
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: FLAGS.dropout})
        if i % 100 == 0:
            saver.save(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'), global_step=i)

    saver.save(sess, os.path.join(FLAGS.model_dir, 'model.ckpt'))
    print(f'Model saved in path: {FLAGS.model_dir}')

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='Use fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000, help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')
    parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/mnt/data', help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/mnt/logs', help='Summaries log directory')
    parser.add_argument('--model_dir', type=str, default='/mnt/model', help='Directory for saving the trained model')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
