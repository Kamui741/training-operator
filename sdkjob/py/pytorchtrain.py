import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import horovod.torch as hvd
    HOROVOD_ENABLED = True
except ImportError:
    HOROVOD_ENABLED = False

FLAGS = None

def check_gpu():
    """自动检测 GPU 可用性"""
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.no_cuda else 'cpu'
    print(f"Using device: {device}")
    return device

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
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.test_batch_size)

    return train_loader, test_loader

class SimpleNN(nn.Module):
    """构建模型"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.dropout = nn.Dropout(1 - FLAGS.dropout)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    if HOROVOD_ENABLED:
        # 初始化 Horovod
        hvd.init()

    device = check_gpu()

    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_data()

    # 创建数据集
    train_loader, test_loader = create_datasets(x_train, y_train, x_test, y_test)

    # 构建模型
    model = SimpleNN().to(device)

    # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = FLAGS.lr * (hvd.size() if HOROVOD_ENABLED else 1)
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr, momentum=FLAGS.momentum)

    if HOROVOD_ENABLED:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    criterion = nn.CrossEntropyLoss()

    # 设置随机种子
    torch.manual_seed(FLAGS.seed)

    # 记录训练开始时间
    start_time = time.time()

    # Train the model.
    for epoch in range(FLAGS.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % FLAGS.log_interval == 0:
                print(f'Epoch {epoch + 1}/{FLAGS.epochs}, Batch {batch_idx}, Loss: {loss.item()}')

    # 评估模型
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}, Accuracy: {correct / len(test_loader.dataset):.4f}')

    # 保存模型
    if not HOROVOD_ENABLED or hvd.rank() == 0:
        torch.save(model.state_dict(), FLAGS.model_dir)

    # 计算并打印训练时长
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in {training_duration:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch MNIST 示例")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="训练的输入批量大小（默认: 64）")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="测试的输入批量大小（默认: 1000）")
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="训练的轮数（默认: 1）")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="学习率（默认: 0.01）")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD 动量（默认: 0.5）")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="随机种子（默认: 1）")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="每训练多少批次记录一次日志（默认: 10）")
    parser.add_argument("--dropout", type=float, default=0.5, metavar="D", help="dropout 概率（默认: 0.5）")
    parser.add_argument("--backend", type=str, choices=["gloo", "nccl", "mpi"], default="nccl", help="分布式训练的后端（gloo、nccl 或 mpi）")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="禁用 CUDA 训练")
    parser.add_argument("--data_dir", required=True, help="数据集目录路径")
    parser.add_argument("--log_dir", required=True, help="保存日志的目录路径")
    parser.add_argument("--model_dir", required=True, help="保存模型的目录路径")

    FLAGS, unparsed = parser.parse_known_args()

    # 使用 --backend 参数来配置分布式后端
    if FLAGS.backend:
        torch.distributed.init_process_group(backend=FLAGS.backend)

    main()
