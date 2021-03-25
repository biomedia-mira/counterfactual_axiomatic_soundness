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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(net: torch.nn.Module,
          device: torch.device,
          trainloader: torch.utils.data.DataLoader,
          testloader: torch.utils.data.DataLoader,
          num_epochs: int,
          path: Path):
    print(device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        cross_entropy = 0.
        self_entropy = 0.
        correct = 0
        total = 0
        avg_img = 0.
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # avg_img += torch.sum(inputs, dim=0)
            cross_entropy_loss = F.cross_entropy(outputs, labels, reduction='mean')
            self_entropy_loss = -torch.mean(torch.sum(F.softmax(outputs, dim=-1) * F.log_softmax(outputs, dim=-1), dim=-1))
            loss = cross_entropy_loss - self_entropy_loss
            loss.backward()
            optimizer.step()
            
            # print statistics
            cross_entropy += cross_entropy_loss.item() / math.log(10.) * labels.size(0)
            self_entropy += self_entropy_loss.item() / math.log(10.) * labels.size(0)
            # p = torch.softmax(outputs, dim=-1)
            # entropy += -torch.sum(p * torch.log(p + 1e-10) / math.log(10.)).item()

            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        if epoch % 50 == 0 and epoch > 0:
            test(net, device, testloader)
            torch.save(net.state_dict(), path)

        print(f'[{epoch + 1:d}, {num_epochs:d}] cross_entropy: {cross_entropy / total:.4f}, '
              f'accuracy: {100. * correct / total:.2f}, '
              f'self entropy: {self_entropy / total:.4f}')
        #imshow((avg_img / total).cpu())
    print('Finished Training')
    torch.save(net.state_dict(), path)


def test(net: torch.nn.Module,
         device: torch.device,
         testloader: torch.utils.data.DataLoader):
    net.to(device)
    correct = 0
    total = 0
    codes = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            codes += list(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test accuracy: {100. * correct / total:.2f}')

    for bits, dtype in zip([32, 16, 8], [torch.float32, torch.float16, torch.uint8]):
        codes_bytes = np.stack([code.type(dtype).cpu().numpy().tobytes() for code in codes])
        num_unique_codes = np.unique(codes_bytes).size
        print(f'Number of unique codes with {bits:2d} bits: {num_unique_codes:2}/{codes_bytes.size:d} = '
              f'{100. * num_unique_codes / codes_bytes.size:.2f}%')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=5)

testset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=5)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
net = Net()

path = Path('/tmp/cifar10_model.pt')
overwrite = True

if not path.exists() or overwrite:
    train(net, device, trainloader, testloader, 300, path)
net.load_state_dict(torch.load(path))
codes = test(net, device, trainloader)
codes = test(net, device, testloader)
