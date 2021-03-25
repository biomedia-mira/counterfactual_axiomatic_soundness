import math
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Subset
from torch.utils.data.dataloader import DataLoader
import torch


def show_matrix(matrix: torch.Tensor, title: str):
    offset = mcolors.TwoSlopeNorm(vcenter=0.)
    plt.imshow(offset(matrix), cmap='seismic')
    plt.title(title)
    plt.show()


def cifar():
    transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float64)])
    trainset = torchvision.datasets.CIFAR10('/tmp', train=True, download=True, transform=transform)
    trainset = Subset(trainset, indices=range(10000))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10('/tmp', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=True, num_workers=0)
    return trainloader, testloader


def mnist():
    path = '/vol/biomedic2/np716/data/mnist/'
    transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float64)])
    trainset = torchvision.datasets.MNIST(path, train=True, download=False, transform=transform)
    trainset = Subset(trainset, indices=range(10000))
    trainloader = DataLoader(trainset, batch_size=24, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(path, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=48, shuffle=False, num_workers=0)
    return trainloader, testloader


def kde(x: torch.Tensor,
        dataloader: DataLoader,
        covariance_matrix):
    precision_matrix = torch.linalg.inv(covariance_matrix)
    determinant = torch.linalg.det(covariance_matrix)

    # h = ((4 / (p + 2)) ** (2 / (p + 4)) * n ** (-2 / (p + 4)))

    x = x.reshape(x.shape[0], -1)
    values = []
    for mean, _ in dataloader:
        t = (x.unsqueeze(1) - mean.reshape(mean.shape[0], -1).unsqueeze(0)).unsqueeze(-2)
        values.append(-.5 * (t @ precision_matrix @ t.transpose(3, 2)).reshape(t.shape[:2]))

    log_densities = torch.cat(values, dim=1) - .5 * (log_determinant + rank * math.log(2 * math.pi))
    log_density = torch.logsumexp(log_densities, dim=1) - math.log(n)
    return log_density


trainloader, testloader = mnist()
kernel_size = (3, 3)
num_channels = 1
dim = np.prod(kernel_size) * num_channels

mean = torch.zeros(dim)
size = 0
for x, _ in trainloader:
    patches = F.unfold(x, kernel_size=kernel_size).movedim(2, 0).flatten(0, 1)
    mean += torch.sum(patches, dim=0)
    size += len(patches)
mean /= size
# plt.imshow(mean.reshape(num_channels, *kernel_size).permute(1, 2, 0), cmap='gray')
# plt.show()

plt.imshow((255 * mean.reshape(num_channels, *kernel_size)).type(torch.int).permute(1, 2, 0), cmap='gray')
plt.show()

covariance_matrix = torch.zeros(dim, dim, dtype=torch.float64)
for x, _ in trainloader:
    patches = F.unfold(x, kernel_size=kernel_size).movedim(2, 0).flatten(0, 1)
    t = patches - mean
    covariance_matrix += t.T @ t
covariance_matrix /= size
show_matrix(covariance_matrix, 'Cov')

# test markov assumption


exit(1)
