import matplotlib.pyplot as plt
import numpy as np
import torch.autograd
import torchvision
import torchvision.transforms as transforms
import matplotlib.colors as mcolors
import math
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset


def show_matrix(matrix: torch.Tensor, title: str):
    offset = mcolors.TwoSlopeNorm(vcenter=0.)
    plt.imshow(offset(matrix), cmap='seismic')
    plt.title(title)
    plt.show()


def kde(x: torch.Tensor,
        dataloader: DataLoader,
        v: torch.Tensor,
        s: torch.Tensor,
        u: torch.Tensor,
        rank: Optional[int] = None):
    if rank is None:
        rank = len(s)
    n = len(dataloader.dataset)

    log_determinant = torch.sum(torch.log(s[:rank]))
    v_ = v[:, :rank]
    s_ = s[:rank]
    u_ = u[:, :rank]

    # h = ((4 / (p + 2)) ** (2 / (p + 4)) * n ** (-2 / (p + 4)))

    x = x.reshape(x.shape[0], -1)
    values = []
    for mean, _ in dataloader:
        t = (x.unsqueeze(1) - mean.reshape(mean.shape[0], -1).unsqueeze(0)).unsqueeze(-2)
        values.append(-.5 * (t @ v_ @ torch.diag(1. / s_) @ u_.T @ t.transpose(3, 2)).reshape(
            t.shape[:2]))

    log_densities = torch.cat(values, dim=1) - .5 * (log_determinant + rank * math.log(2 * math.pi))
    log_density = torch.logsumexp(log_densities, dim=1) - math.log(n)
    return log_density


def kde_gen(
        dataloader: DataLoader,
        v: torch.Tensor,
        s: torch.Tensor,
        u: torch.Tensor,
        rank: Optional[int] = None):
    if rank is None:
        rank = len(s)
    n = len(dataloader.dataset)

    log_determinant = torch.sum(torch.log(s[:rank]))
    v_ = v[:, :rank]
    s_ = s[:rank]
    u_ = u[:, :rank]

    # h = ((4 / (p + 2)) ** (2 / (p + 4)) * n ** (-2 / (p + 4)))

    dataiter = iter(trainloader)
    train_images, labels = dataiter.next()

    # x = torch.rand_like(train_images[0], requires_grad=True)
    # # x = torch.tensor(train_images, requires_grad=True)
    # optimizer = optim.SGD([x], lr=10000, momentum=0.)
    #
    # x = x.reshape(1, -1)
    # shape = [1, *train_images.shape[1:]]
    # plt.imshow(np.transpose(torchvision.utils.make_grid(x.reshape(shape)).detach().numpy(), (1, 2, 0)))
    # plt.show()

    # for i in range(1000):
    #     accumulate_loss = 0.
    #     optimizer.zero_grad()
    #     for mean, _ in dataloader:
    #         t = (x.unsqueeze(1) - mean.reshape(mean.shape[0], -1).unsqueeze(0)).unsqueeze(-2)
    #         values = -.5 * (t @ v_ @ torch.diag(1. / s_) @ u_.T @ t.transpose(3, 2)).reshape(t.shape[:2])
    #         loss = -torch.sum(torch.exp(values))
    #         # loss = -torch.sum(torch.logsumexp(values, dim=1))
    #         accumulate_loss += loss.item()
    #         loss.backward(retain_graph=True)
    #     print(i, accumulate_loss)
    #     optimizer.step()
    #     if i % 2 == 0:
    #         plt.imshow(
    #             np.transpose(torchvision.utils.make_grid(x.reshape(shape)).detach().numpy(), (1, 2, 0)))
    #         plt.show()

    x = torch.rand_like(train_images)
    # x = torch.tensor(train_images)
    x = x.reshape(train_images.shape[0], -1)
    shape = train_images.shape
    plt.imshow(np.transpose(torchvision.utils.make_grid(x.reshape(shape)).detach().numpy(), (1, 2, 0)))
    plt.show()

    for i in range(1000):
        p_x_ = []
        p_x_prime = []
        for mean, _ in dataloader:
            # calculate distance x - mu and it's transpose
            d = (x.unsqueeze(1) - mean.reshape(mean.shape[0], -1).unsqueeze(0)).unsqueeze(-2)
            d_transpose = d.transpose(3, 2)
            # calculate the term inside the multivariate normal exponential
            a_ = -.5 * ((d @ v_) @ torch.diag(1. / s_) @ (u_.T @ d_transpose)).squeeze(-1).squeeze(-1)
            p_x_.append(a_)
            b_ = (v_ @ torch.diag(1. / s_) @ (u_.T @ d_transpose)).squeeze(-1)
            p_x_prime.append(b_)

        log_densities = torch.cat(p_x_, dim=1) - .5 * (log_determinant + rank * math.log(2 * math.pi))

        derivative_ = torch.mean(-torch.exp(log_densities).unsqueeze(-1) * torch.cat(p_x_prime, dim=1), dim=1)
        #x = x + .1 * derivative_ / torch.norm(derivative_, dim=1, keepdim=True, p=2)
        x = x + 1. * derivative_

        log_density = torch.logsumexp(log_densities, dim=1) - math.log(n)
        print(torch.mean(log_density))
        if i % 2 == 0:
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(x.reshape(shape)).detach().numpy(), (1, 2, 0)))
            plt.show()

    return 1


# Load and transform data
transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float64)])
# trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
# trainset = Subset(trainset, indices=range(10000))
# trainloader = DataLoader(trainset, batch_size=24, shuffle=True, num_workers=0)
# testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=48, shuffle=False, num_workers=0)

trainset = torchvision.datasets.CIFAR10('/tmp', train=True, download=True, transform=transform)
trainset = Subset(trainset, indices=range(10000))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10('/tmp', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=True, num_workers=0)


dataiter = iter(trainloader)
images, labels = dataiter.next()

print('Labels: ', labels)
print('Batch shape: ', images.size())
im = torchvision.utils.make_grid(images)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

shape = images.shape[1:]
dim = np.prod(shape)

# calculate mean
mean = torch.zeros(dim)
for x, _ in trainloader:
    mean += torch.sum(x.reshape(x.shape[0], -1), dim=0)
mean /= len(trainset)
# plt.imshow(mean.reshape(shape).permute(1, 2, 0), cmap='gray')
# plt.show()

# calculate sample covariance
covariance_matrix = torch.zeros(dim, dim, dtype=torch.float64)
for x, _ in trainloader:
    t = x.reshape(x.shape[0], -1) - mean
    covariance_matrix += t.T @ t
covariance_matrix /= len(trainset)
show_matrix(covariance_matrix, 'Covariance Matrix')
# assert sample covariance is positive semi definite
u, s, v = torch.svd(covariance_matrix)
assert all(s > 0)
log_determinant = torch.sum(torch.log(s))
print(f'Log-determinant = {float(log_determinant):.4f}')
error_map = torch.pow((u @ torch.diag(s) @ v.T) - covariance_matrix, 2)
plt.imshow(error_map, cmap='Reds')
plt.show()

# compute precision matrix
precision_matrix = v @ torch.diag(1. / s) @ u.T
show_matrix(precision_matrix, 'Precision Matrix')
show_matrix(covariance_matrix @ precision_matrix, 'Cov @ Pres')

# low-rank factorisation
threshold = .75
rank = int(torch.where(torch.cumsum(s, dim=0) > threshold * torch.sum(s))[0][0])
print(f'Threshold: {threshold:.2f} ; Rank = {rank:d}')
low_rank_sigma = (u[:, :rank] @ torch.diag(s[:rank]) @ v[:, :rank].T)
low_rank_precision_matrix = v[:, :rank] @ torch.diag(1. / s[:rank]) @ u[:, :rank].T
show_matrix(low_rank_sigma, 'Low-rank Covariance Matrix')
show_matrix(low_rank_precision_matrix, 'Low-rank Precision Matrix')
show_matrix(low_rank_sigma @ low_rank_precision_matrix, '(Low-rank) Cov @ Pres')
# low-rank KDE density
dataiter = iter(trainloader)
train_images, labels = dataiter.next()
dataiter = iter(testloader)
test_images, labels = dataiter.next()
print(kde(train_images, trainloader, v, s, u))
kde_gen(trainloader, v, s, u)

# print(kde(torch.rand_like(train_images), trainloader, v, s, u))
# print(kde(test_images, trainloader, v, s, u))
# print(kde(1 - test_images, trainloader, v, s, u))
