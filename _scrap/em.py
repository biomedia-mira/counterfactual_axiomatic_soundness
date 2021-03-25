from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18


def show_matrix(matrix: torch.Tensor, title: str):
    offset = mcolors.TwoSlopeNorm(vcenter=0.)
    plt.imshow(offset(matrix), cmap='seismic')
    plt.title(title)
    plt.show()


# class Net(nn.Module):
#     def __init__(self, num_clusters: int = 10):
#         super(Net, self).__init__()
#         self.num_clusters = num_clusters
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_clusters ** 2)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x.reshape(-1, *(self.num_clusters, self.num_clusters))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(device: torch.device,
          trainloader: torch.utils.data.DataLoader,
          num_epochs: int,
          path: Path):
    print(device)
    writer = SummaryWriter('/tmp/mnist_test_em')
    num_clusters = 10
    num_labels = 10
    net = resnet18(num_classes=num_clusters ** 2)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.0)
    current_clusters_all = torch.randint(0, num_clusters, (len(trainloader.dataset),), device=device)
    batch_size = 128
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        cross_entropy = 0.
        correct = 0
        total = 0
        cm = torch.zeros((num_labels, num_clusters), device=device)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            if inputs.shape[1] == 1:
                inputs = inputs.repeat((1, 3, 1, 1))
            for e in range(10):
                current_clusters = current_clusters_all[i * batch_size:(i + 1) * batch_size]
                optimizer.zero_grad()
                cluster_transition = net(inputs)
                cluster_transition = cluster_transition.reshape(-1, *(num_clusters, num_clusters))
                cluster_assignments = cluster_transition[torch.arange(cluster_transition.shape[0]), current_clusters]
                new_clusters = torch.argmax(cluster_assignments, dim=-1)
                loss = -torch.sum(F.log_softmax(cluster_assignments, dim=-1))
                cross_entropy += loss.item()
                loss.backward()
                optimizer.step()
                current_clusters_all[i * batch_size:(i + 1) * batch_size] = new_clusters
            for c in range(num_clusters):
                p = inputs[current_clusters == c][:8]
                if len(p > 0):
                    writer.add_image(f'c_{c:d}', torchvision.utils.make_grid(p, normalize=True), global_step=epoch)

            for l in range(num_labels):
                for c in range(num_clusters):
                    cm[l, c] += torch.sum(torch.logical_and(current_clusters == c, labels == l))
            total += labels.size(0)
        # show_matrix(cm.detach().cpu().numpy(), 'CM')

        cm_ = cm.detach().cpu().numpy()
        cm_ = 100. * cm_ / np.sum(cm_, axis=1, keepdims=True)
        # plt.matshow(cm_)
        fig, ax = plt.subplots()
        ax.imshow(cm_)

        for i in range(num_labels):
            for j in range(num_clusters):
                ax.text(j, i, f'{cm_[i, j]:.1f}', ha='center', va='center')
        fig.tight_layout()
        plt.show()

        if epoch % 2 == 0 and epoch > 0:
            torch.save(net.state_dict(), path)

        print(f'[{epoch + 1:d}, {num_epochs:d}] cross_entropy: {cross_entropy / total:.4e}, '
              f'accuracy: {100. * correct / total:.2f}')
    print('Finished Training')
    torch.save(net.state_dict(), path)


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
#                                           shuffle=False, num_workers=0, drop_last=True)
#
# testset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                          shuffle=False, num_workers=0)

path = '/vol/biomedic2/np716/data/mnist/'
trainset = torchvision.datasets.MNIST(path, train=True, download=False, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(path, train=False, download=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=48, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

path = Path('/tmp/cifar10_model.pt')
overwrite = True

if not path.exists() or overwrite:
    train(device, trainloader, 300, path)
