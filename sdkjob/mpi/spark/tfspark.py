'''
Author: ChZheng
Date: 2024-09-23 13:59:51
LastEditTime: 2024-09-23 13:59:54
LastEditors: ChZheng
Description:
FilePath: /horovod/Users/apple/go/src/github.com/training-operator/sdkjob/mpi/spark/tfspark.py
'''
import argparse
import os
import sys

from pyspark import SparkConf
from pyspark.sql import SparkSession

from horovod.spark import run
from horovod.tensorflow.data.compute_service import TfDataServiceConfig
from tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher import train_fn as train_fn_compute_side
from tensorflow2_mnist_data_service_train_fn_training_side_dispatcher import train_fn as train_fn_training_side

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# This exemplifies how to use the Tensorflow Compute Service with Horovod.
# The Tensorflow Dispatcher can reside with the training script, or the compute service.
# If you use only one of these options, you can ignore the respective code of the other option in this example.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("configfile", type=str,
                        help="The path to the compute service config file.")

    parser.add_argument("--reuse-dataset", required=False, action="store_true", default=False,
                        help="Reusing the dataset allows the training tasks to read from a single dataset "
                             "in a first-come-first-serve manner.",
                        dest="reuse_dataset")

    parser.add_argument("--round-robin", required=False, action="store_true", default=False,
                        help="Reusing the dataset can be done round-robin instead of first-come-first-serve.",
                        dest="round_robin")

    # 新增 data_dir, model_dir, log_dir 参数
    parser.add_argument("--data-dir", type=str, default="/tmp/data",
                        help="Directory to load the MNIST dataset.")
    parser.add_argument("--model-dir", type=str, default="/tmp/model",
                        help="Directory to save the trained model.")
    parser.add_argument("--log-dir", type=str, default="/tmp/logs",
                        help="Directory to save the logs.")

    parsed_args = parser.parse_args()

    # 读取 compute service 配置
    compute_config = TfDataServiceConfig.read(parsed_args.configfile, wait_for_file_creation=True)

    # 创建 Spark Session
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark_context = spark.sparkContext
    training_tasks = spark_context.defaultParallelism

    if compute_config.dispatchers > 1 and training_tasks != compute_config.dispatchers:
        print(f'The number of training tasks ({training_tasks}) must match '
              f'the number of dispatchers ({compute_config.dispatchers}) configured in the '
              f'data service config file ({parsed_args.configfile}).', file=sys.stderr)
        sys.exit(1)

    # 选择正确的 train_fn 取决于 dispatcher 位置
    if compute_config.dispatcher_side == 'training':
        train_fn = train_fn_training_side
    elif compute_config.dispatcher_side == 'compute':
        train_fn = train_fn_compute_side
    else:
        raise ValueError(f'Unsupported dispatcher side: {compute_config.dispatcher_side}')

    # 打印数据集、模型和日志目录路径
    print(f'Data Directory: {parsed_args.data_dir}')
    print(f'Model Directory: {parsed_args.model_dir}')
    print(f'Log Directory: {parsed_args.log_dir}')

    # 运行分布式训练
    run(train_fn,
        args=(compute_config,),
        kwargs={
            'reuse_dataset': parsed_args.reuse_dataset,
            'round_robin': parsed_args.round_robin,
            'data_dir': parsed_args.data_dir,
            'model_dir': parsed_args.model_dir,
            'log_dir': parsed_args.log_dir
        },
        num_proc=training_tasks,
        stdout=sys.stdout,
        stderr=sys.stderr)

    compute = compute_config.compute_client(verbose=2)
    compute.shutdown()
