from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.functional

from datasets.confounded_dataset import Mechanism

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


def show_cm(cm: np.ndarray):
    print(f'Uncertainty coefficient {calculate_uncertainty_coefficient(cm, epsilon=1e-8):.4f}')
    fig, ax = plt.subplots()
    ax.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha="center", va="center")
    fig.show()


class Colorize(Mechanism):
    def __init__(self, cm: np.ndarray, labels: np.ndarray):
        self.color = np.array(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), labels)))
        self.colorize = torch.tensor(COLORS)

    def __call__(self, index: int, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        color = self.color[index]
        return torch.cat([image] * 3, dim=0) * self.colorize[color].view(3, 1, 1), {'color': torch.tensor(color)}
