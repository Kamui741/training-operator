import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

try:
    import horovod.torch as hvd
    HOROVOD_ENABLED = True
except ImportError:
    HOROVOD_ENABLED = False

class SimpleNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def check_gpu(use_cuda):
    if torch.cuda.is_available() and use_cuda:
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available, using CPU.")

def load_data(data_dir):
    """从本地加载 MNIST 数据集"""
    with np.load(os.path.join(data_dir, 'mnist.npz')) as data:
        x_train = torch.tensor(data['x_train'].reshape(-1, 784) / 255.0, dtype=torch.float32)
        y_train = torch.tensor(data['y_train'], dtype=torch.long)
        x_test = torch.tensor(data['x_test'].reshape(-1, 784) / 255.0, dtype=torch.float32)
        y_test = torch.tensor(data['y_test'], dtype=torch.long)

    return (x_train, y_train), (x_test, y_test)

def create_datasets(x_train, y_train, x_test, y_test, batch_size, test_batch_size, use_horovod):
    """创建训练和测试数据集"""
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if use_horovod else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--backend', type=str, default="nccl")
    parser.add_argument('--use-horovod', action='store_true', default=False)
    parser.add_argument('--use-mixed-precision', action='store_true', default=False)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if HOROVOD_ENABLED and args.use_horovod:
        hvd.init()
        if use_cuda:
            torch.cuda.set_device(hvd.local_rank())
        torch.distributed.init_process_group(backend=args.backend)

    device = torch.device("cuda" if use_cuda else "cpu")
    check_gpu(use_cuda)

    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_data(args.data_dir)
    train_loader, test_loader = create_datasets(x_train, y_train, x_test, y_test, args.batch_size, args.test_batch_size, args.use_horovod)

    model = SimpleNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if HOROVOD_ENABLED and args.use_horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0 and (not HOROVOD_ENABLED or hvd.rank() == 0):
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                correct += (output.argmax(1) == target).sum().item()

        test_loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)
        if not HOROVOD_ENABLED or hvd.rank() == 0:
            print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    if args.save_model and (not HOROVOD_ENABLED or hvd.rank() == 0):
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
