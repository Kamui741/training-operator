import argparse
import os
import time
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
import horovod
import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                    help='use mixed precision for training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir', required=True,
                    help='location of the mnist.npz dataset')
parser.add_argument('--model-dir', required=True,
                    help='directory to save the model')
parser.add_argument('--log-dir', required=True,
                    help='directory to save training logs')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def main(args):
    def train_epoch(epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_sampler)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        end_time = time.time()
        print(f"Epoch {epoch} training time: {end_time - start_time:.2f} seconds")

    def test():
        model.eval()
        test_loss = 0.
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * correct / len(test_loader.dataset):.0f}%)')

    hvd.init()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    # Load the mnist.npz dataset
    data = np.load(args.data_dir)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train).unsqueeze(1).float(), torch.tensor(y_train).long())
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_test).unsqueeze(1).float(), torch.tensor(y_test).long())

    # Use DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    model = Net()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch)
        test()

    # Save the model
    if hvd.rank() == 0:
        model_path = os.path.join(args.model_dir, "mnist_cnn.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
