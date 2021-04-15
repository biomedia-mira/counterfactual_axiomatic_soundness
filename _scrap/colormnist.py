from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.functional
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

COLORS = ((1, 0, 0),
          (0, 1, 0),
          (0, 0, 1),
          (1, 1, 0),
          (1, 0, 1),
          (0, 1, 1),
          (1, 1, 1),
          (.5, 0, 0),
          (0, .5, 0),
          (0, 0, .5))


class Mechanism(ABC):
    @abstractmethod
    def __call__(self, index: int, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


def get_uniform_class_conditioned_color_distribution(num_classes: int, num_colors: int):
    return np.ones((num_classes, num_colors)) / num_colors


def get_random_class_conditioned_color_distribution(num_classes: int, num_colors: int, temperature: float = .1,
                                                    seed: int = 1):
    random_state = np.random.RandomState(seed=seed)
    logits = random_state.random(size=(num_classes, num_colors))
    tmp = np.exp(logits / temperature)
    return tmp / tmp.sum(1, keepdims=True)


def get_diagonal_class_conditioned_color_distribution(num_classes: int, num_colors: int, noise: float = 0.):
    assert num_classes == num_colors
    return (np.eye(num_classes) * (1. - noise)) + (
            np.ones((num_classes, num_classes)) - np.eye(num_classes)) * noise / (num_classes - 1)


def calculate_uncertainty_coefficient(cm: np.ndarray, epsilon: float = 0.):
    # U(X|Y) = (H(X) - H(X|Y)) / H(X)
    # assumes uniform class distribution
    marginal_x = cm.sum(0) / cm.sum()
    h_x = -np.sum(marginal_x * np.log(marginal_x + epsilon))
    h_x_given_y = -np.sum((1. / cm.shape[0]) * cm * np.log(cm + epsilon))
    return (h_x - h_x_given_y) / h_x


class Colorize(Mechanism):
    def __init__(self, cm: np.ndarray):
        self.cm = cm
        fig, ax = plt.subplots()
        print(f'Uncertainty coefficient {calculate_uncertainty_coefficient(cm, epsilon=1e-8):.4f}')
        ax.imshow(self.cm)
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(j, i, f'{self.cm[i, j]:.2f}', ha="center", va="center")
        fig.show()
        self.colors = torch.tensor(COLORS)

    def __call__(self, index: int, image: torch.Tensor, label: torch.Tensor):
        random_state = np.random.RandomState(index)
        color_idx = random_state.choice(self.cm.shape[1], p=self.cm[label])
        return torch.cat([image] * 3, dim=0) * self.colors[color_idx].view(3, 1, 1)


class CounfoundedMNIST(Dataset):
    def __init__(self, root: str, mechanisms: Sequence[Mechanism], train: bool = True, download: bool = False) -> None:
        self.base_dataset = torchvision.datasets.MNIST(root=root, train=train, download=download,
                                                       transform=transforms.ToTensor())
        self.mechanisms = mechanisms
        self.train = train

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.base_dataset.__getitem__(index)
        for mechanism in self.mechanisms:
            image = mechanism(index, image, label)

        return image, label

    def __len__(self):
        return len(self.base_dataset)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('GPU is not available!')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    print(device)
    path = '../data/mnist/'
    batch_size = 24
    num_workers = 0
    uniform_cm = get_uniform_class_conditioned_color_distribution(10, 10)
    diagonal_cm = get_diagonal_class_conditioned_color_distribution(10, 10, noise=.05)
    trainset = CounfoundedMNIST(path, mechanisms=[Colorize(diagonal_cm)], train=True, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = CounfoundedMNIST(path, mechanisms=[Colorize(uniform_cm)], train=False, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    im = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()
