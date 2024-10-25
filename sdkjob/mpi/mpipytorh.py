import argparse
import os
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import transforms
import horovod
import horovod.torch as hvd
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类，用于加载 mnist.npz
class MNISTDataset(Dataset):
    def __init__(self, data_dir):
        with np.load(os.path.join(data_dir, 'mnist.npz')) as data:
            self.x = data['x_train']
            self.y = data['y_train']
        self.x = self.x.astype(np.float32) / 255.0  # 归一化
        self.x = np.expand_dims(self.x, axis=1)  # 增加通道维度

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--data-dir', type=str, default='./data',
                    help='location of the training dataset in the local filesystem')
parser.add_argument('--model-dir', type=str, default='./model',
                    help='directory to save the trained model')
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

# Arguments when not run through horovodrun
parser.add_argument('--num-proc', type=int)
parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')

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
    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # 加载本地 MNIST 数据集
    data_dir = args.data_dir
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = MNISTDataset(data_dir)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, **kwargs)

    # 测试集数据集
    test_dataset = MNISTDataset(data_dir)  # 假设测试集也是在同一文件中
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             sampler=test_sampler, **kwargs)

    model = Net()

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Horovod: 广播模型参数
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_sampler)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 测试过程
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_accuracy += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        if hvd.rank() == 0:
            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100. * test_accuracy:.2f}%\n')

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.num_proc:
        horovod.run(main,
                    args=(args,),
                    np=args.num_proc,
                    hosts=args.hosts,
                    use_gloo=args.communication == 'gloo',
                    use_mpi=args.communication == 'mpi')
    else:
        main(args)
