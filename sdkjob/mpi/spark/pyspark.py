'''
Author: ChZheng
Date: 2024-09-23 13:59:26
LastEditTime: 2024-09-23 13:59:29
LastEditors: ChZheng
Description:
FilePath: /horovod/Users/apple/go/src/github.com/training-operator/sdkjob/mpi/spark/pyspark.py
'''
import argparse
import os
import subprocess
import sys
import numpy as np

import pyspark
import pyspark.sql.types as T
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import horovod.spark.torch as hvd
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store


# 判断pyspark版本并引入适合的OneHotEncoder
if version.parse(pyspark.__version__) < version.parse('3.0.0'):
    from pyspark.ml.feature import OneHotEncoderEstimator as OneHotEncoder
else:
    from pyspark.ml.feature import OneHotEncoder


parser = argparse.ArgumentParser(description='PyTorch Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master', help='spark master to connect to')
parser.add_argument('--num-proc', type=int, help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=12, help='number of epochs to train')
parser.add_argument('--work-dir', default='/tmp', help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--data-dir', default='/tmp', help='location of the training dataset')
parser.add_argument('--model-dir', default='/tmp', help='directory to save the trained model')
parser.add_argument('--log-dir', default='/tmp', help='directory to save logs')
parser.add_argument('--backward-passes-per-step', type=int, default=1, help='number of backward passes to perform before calling hvd.allreduce')

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize SparkSession
    conf = SparkConf().setAppName('pytorch_spark_mnist').set('spark.sql.shuffle.partitions', '16')
    if args.master:
        conf.setMaster(args.master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Setup our store for intermediate data
    store = Store.create(args.work_dir)

    # Load MNIST dataset from the local file (mnist.npz)
    mnist_npz_path = os.path.join(args.data_dir, 'mnist.npz')
    if not os.path.exists(mnist_npz_path):
        raise FileNotFoundError(f"MNIST dataset not found at {mnist_npz_path}")

    with np.load(mnist_npz_path) as data:
        X_train, y_train = data['x_train'], data['y_train']
        X_test, y_test = data['x_test'], data['y_test']

    # Convert data into Spark DataFrame
    train_data = [(float(y), X.reshape(28 * 28).tolist()) for X, y in zip(X_train, y_train)]
    test_data = [(float(y), X.reshape(28 * 28).tolist()) for X, y in zip(X_test, y_test)]

    train_df = spark.createDataFrame(train_data, ['label', 'features'])
    test_df = spark.createDataFrame(test_data, ['label', 'features'])

    # One-hot encode labels into SparseVectors
    encoder = OneHotEncoder(inputCols=['label'], outputCols=['label_vec'], dropLast=False)
    model = encoder.fit(train_df)
    train_df = model.transform(train_df)

    # Define the PyTorch model without any Horovod-specific parameters
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, features):
            x = features.float()
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss = nn.NLLLoss()

    # Train a Horovod Spark Estimator on the DataFrame
    backend = SparkBackend(num_proc=args.num_proc, stdout=sys.stdout, stderr=sys.stderr, prefix_output_with_timestamp=True)
    torch_estimator = hvd.TorchEstimator(backend=backend,
                                         store=store,
                                         model=model,
                                         optimizer=optimizer,
                                         loss=lambda input, target: loss(input, target.long()),
                                         input_shapes=[[-1, 1, 28, 28]],
                                         feature_cols=['features'],
                                         label_cols=['label'],
                                         batch_size=args.batch_size,
                                         epochs=args.epochs,
                                         validation=0.1,
                                         backward_passes_per_step=args.backward_passes_per_step,
                                         verbose=1)

    torch_model = torch_estimator.fit(train_df).setOutputCols(['label_prob'])

    # Evaluate the model on the held-out test DataFrame
    pred_df = torch_model.transform(test_df)

    argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
    pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
    evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
    accuracy = evaluator.evaluate(pred_df)
    print('Test accuracy:', accuracy)

    # Save the trained model
    model_save_path = os.path.join(args.model_dir, 'pytorch_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Write logs
    log_path = os.path.join(args.log_dir, 'training.log')
    with open(log_path, 'w') as log_file:
        log_file.write(f'Training completed.\nTest accuracy: {accuracy}\n')

    print(f"Logs saved to {log_path}")

    spark.stop()
