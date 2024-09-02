from __future__ import print_function
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        mnist_data = np.load(os.path.join(data_dir, 'mnist.npz'))
        self.transform = transform
        if train:
            self.data = mnist_data['x_train']
            self.targets = mnist_data['y_train']
        else:
            self.data = mnist_data['x_test']
            self.targets = mnist_data['y_test']
        self.data = self.data[:, :, :, None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, target

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
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tloss={loss.item():.4f}")
            writer.add_scalar("loss", loss.item(), epoch * len(train_loader) + batch_idx)

def test(model, device, test_loader, writer, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"\naccuracy={accuracy:.4f}\n")
    writer.add_scalar("accuracy", accuracy, epoch)

def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument("--dir", default="logs", metavar="L", help="directory where summary logs are stored")
    parser.add_argument("--backend", type=str, help="Distributed backend", choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI], default=dist.Backend.GLOO)
    parser.add_argument("--data-dir", type=str, default="/mnt/data", help="directory where MNIST dataset is stored")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(args.dir)
    torch.manual_seed(args.seed)
    model = Net().to(device)

    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    train_ds = MNISTDataset(data_dir=args.data_dir, train=True, transform=transforms.ToTensor())
    test_ds = MNISTDataset(data_dir=args.data_dir, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=DistributedSampler(train_ds))
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, sampler=DistributedSampler(test_ds))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer)
        test(model, device, test_loader, writer, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":
    main()
