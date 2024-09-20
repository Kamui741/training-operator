from __future__ import print_function

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, epoch, writer):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)


def test(model, device, test_loader, writer, epoch):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = float(correct) / len(test_loader.dataset)
    print("\naccuracy={:.4f}\n".format(accuracy))
    writer.add_scalar("accuracy", accuracy, epoch)


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST 示例")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="训练的输入批量大小（默认: 64）")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="测试的输入批量大小（默认: 1000）")
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="训练的轮数（默认: 1）")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="学习率（默认: 0.01）")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD 动量（默认: 0.5）")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="禁用 CUDA 训练")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="随机种子（默认: 1）")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="每训练多少批次记录一次日志（默认: 10）")
    parser.add_argument("--save-model", action="store_true", default=False, help="是否保存当前模型")
    parser.add_argument("--dir", default="logs", metavar="L", help="保存日志的目录")
    parser.add_argument("--backend", type=str, choices=["gloo", "nccl", "mpi"], default="mpi", help="分布式训练的后端（gloo、nccl 或 mpi）")
    parser.add_argument("--data-dir", required=True, help="数据集目录路径")
    parser.add_argument("--log-dir", required=True, help="保存日志的目录路径")
    parser.add_argument("--model-dir", required=True, help="保存模型的目录路径")


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != "nccl":
            print("Warning: Using 'nccl' backend is recommended for GPU.")

    writer = SummaryWriter(args.log_dir)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)

    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, sampler=DistributedSampler(train_ds))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size, sampler=DistributedSampler(test_ds))

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer)
        test(model, device, test_loader, writer, epoch)
    end_time = time.time()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.model_dir, "mnist_cnn.pt"))

    # Log training duration
    duration = end_time - start_time
    print(f"Training duration: {duration:.2f} seconds")
    writer.add_text("Training Duration", f"Training duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
