import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil
torch.autograd.set_detect_anomaly(True)
NUM_WORKERS = 0


class Net(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 10):
        super(Net, self).__init__()
        self.conv00 = nn.Conv2d(in_channels, 6, 3, padding=1)
        self.conv0 = nn.Conv2d(6, 6, 3, padding=1)
        self.conv1 = nn.Conv2d(6, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channels)

    def forward(self, x):
        x = F.relu(self.conv0(F.relu(self.conv00(x))))
        f1 = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, f1


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(net: torch.nn.Module,
          device: torch.device,
          trainloader: DataLoader,
          testloader: DataLoader,
          num_epochs: int,
          job_dir: Path):
    job_dir.mkdir(exist_ok=True, parents=True)
    log_dir = job_dir / 'logs'
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(str(log_dir))
    net.to(device)
    mi_net = Net(9, 1)
    mi_net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer_mi = optim.SGD(mi_net.parameters(), lr=1e-4, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        cross_entropy = 0.
        correct = 0
        total = 0
        mi = 0
        mi_t1 = 0
        mi_t2 = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            optimizer_mi.zero_grad()

            # forward + backward + optimize
            outputs, f1 = net(inputs)
            cross_entropy_loss = F.cross_entropy(outputs, labels, reduction='mean')

            input_ = torch.cat([torch.cat([inputs, f1], dim=1), torch.cat([inputs.roll(1, 0), f1], dim=1)], dim=0)
            output_ = -F.softplus(-mi_net(input_)[0])
            t1, pre_t2 = torch.split(output_, len(output_) // 2, dim=0)
            t2 = -1 - torch.log(-pre_t2)
            t1 = torch.sum(t1)
            t2 = torch.sum(t2)
            mi_loss = t1 - t2

            # minimise cross entropy and minimize mutual information
            loss = cross_entropy_loss + mi_loss
            loss.backward()
            optimizer.step()

            # maximise mutual information
            for p in mi_net.parameters():
                p.grad.data.mul_(-1)

            optimizer_mi.step()

            #print(t1, t2, mi_loss)

            # statistics
            cross_entropy += cross_entropy_loss.item() / math.log(10.) * labels.size(0)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            mi += mi_loss.item() * labels.size(0)
            mi_t1 += t1.item() * labels.size(0)
            mi_t2 += torch.sum(pre_t2).item() * labels.size(0)

        writer.add_scalar('loss', cross_entropy / total, global_step=epoch)
        writer.add_scalar('accuracy', 100. * correct / total, global_step=epoch)
        writer.add_scalar('mi', mi / total, global_step=epoch)
        writer.add_scalar('mi_t1', mi_t1 / total, global_step=epoch)
        writer.add_scalar('mi_t2', mi_t1 / total, global_step=epoch)

        writer.add_image('inputs', torchvision.utils.make_grid(inputs[:8], normalize=True), global_step=epoch)
        writer.add_image('f1', torchvision.utils.make_grid(f1[:8].reshape(-1, 1, *f1.shape[2:]),
                                                           normalize=True, nrow=6), global_step=epoch)

        # if epoch % 50 == 0 and epoch > 0:
        #     test(model, device, testloader)
        #     torch.save(model.state_dict(), job_dir / 'model.pt')
        print(f'[{epoch + 1:d}, {num_epochs:d}] cross_entropy: {cross_entropy / total:.4f} '
              f'accuracy: {100. * correct / total:.2f}')

    print('Finished Training')
    torch.save(net.state_dict(), job_dir / 'model.pt')


def test(net: torch.nn.Module,
         device: torch.device,
         testloader: DataLoader):
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test accuracy: {100. * correct / total:.2f}')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not available!')
    device = torch.device('cuda:0')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

    net = Net()
    job_dir = Path('/tmp/test_run')

    overwrite = True
    if not job_dir.exists() or overwrite:
        train(net, device, trainloader, testloader, 300, job_dir)
