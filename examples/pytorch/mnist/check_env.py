'''
Author: ChZheng
Date: 2024-10-23 16:44:29
LastEditTime: 2024-10-23 16:44:33
LastEditors: ChZheng
Description:
FilePath: /horovod/Users/apple/go/src/github.com/training-operator/examples/pytorch/mnist/check_env.py
'''
import sys

def check_pytorch_environment():
    try:
        import torch
        print("=== PyTorch Environment ===")
        print("PyTorch Version:", torch.__version__)
        print("CUDA Available:", torch.cuda.is_available())
        print("CUDA Version:", torch.version.cuda)
        print("cuDNN Version:", torch.backends.cudnn.version())

        if torch.cuda.is_available():
            print("Number of GPUs:", torch.cuda.device_count())
            print("GPU Name:", torch.cuda.get_device_name(0))
            print("Current GPU Device ID:", torch.cuda.current_device())
            print("GPU Memory Allocated:", torch.cuda.memory_allocated())
            print("GPU Memory Cached:", torch.cuda.memory_reserved())
        else:
            print("No GPU available for PyTorch.")
    except ImportError:
        print("PyTorch is not installed or failed to import.")
    except Exception as e:
        print(f"Error checking PyTorch environment: {e}")

def check_tensorflow_environment():
    try:
        import tensorflow as tf
        print("\n=== TensorFlow Environment ===")
        print("TensorFlow Version:", tf.__version__)
        print("CUDA Version:", tf.sysconfig.get_build_info()["cuda_version"])
        print("cuDNN Version:", tf.sysconfig.get_build_info()["cudnn_version"])

        gpus = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available:", len(gpus))

        for gpu in gpus:
            print(f"GPU Name: {gpu}")
            print(f"Memory Growth Enabled: {tf.config.experimental.get_memory_growth(gpu)}")
            print(f"Device Details: {tf.config.experimental.get_device_details(gpu)}")

        if len(gpus) == 0:
            print("No GPU available for TensorFlow.")
    except ImportError:
        print("TensorFlow is not installed or failed to import.")
    except Exception as e:
        print(f"Error checking TensorFlow environment: {e}")

def main():
    try:
        check_pytorch_environment()
    except Exception as e:
        print(f"PyTorch check failed: {e}")

    try:
        check_tensorflow_environment()
    except Exception as e:
        print(f"TensorFlow check failed: {e}")

if __name__ == "__main__":
    main()
