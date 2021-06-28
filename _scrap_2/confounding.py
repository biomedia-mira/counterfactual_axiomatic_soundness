from typing import Dict, Tuple, Type, List, Any
from morphomnist.morpho import ImageMorphology
from morphomnist.perturb import Perturbation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def get_uniform_confusion_matrix(num_rows: int, num_columns: int):
    return np.ones((num_rows, num_columns)) / num_columns


def get_random_confusion_matrix(num_rows: int, num_columns: int, temperature: float = .1,
                                seed: int = 1):
    random_state = np.random.RandomState(seed=seed)
    logits = random_state.random(size=(num_rows, num_columns))
    tmp = np.exp(logits / temperature)
    return tmp / tmp.sum(1, keepdims=True)


def get_diagonal_confusion_matrix(num_rows: int, num_columns: int, noise: float = 0.):
    assert num_rows == num_columns
    return (np.eye(num_rows) * (1. - noise)) + \
           (np.ones((num_rows, num_rows)) - np.eye(num_rows)) * noise / (num_rows - 1)


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


def get_colorize_fun(cm: np.ndarray, labels: np.ndarray):
    color_indices = np.array(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), labels)))
    colors = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (.5, 0, 0),
                       (0, .5, 0), (0, 0, .5)))

    def colorize(index: int, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        color_idx = int(color_indices[index])
        return np.stack([image] * 3, axis=0) * colors[color_idx].reshape(3, 1, 1), {'color': color_idx}

    return colorize


def get_perturbation_fun(cm: np.ndarray,
                         labels: np.ndarray,
                         perturbation_name: str,
                         perturbation_class: Type[Perturbation],
                         parameters: List[Dict[str, Any]]):
    indices = np.array(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), labels)))

    def perturbation_fun(index: int, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert image.ndim == 2
        perturbation_idx = int(indices[index])
        image = perturbation_class(**parameters[perturbation_idx])(ImageMorphology(image))
        image = gaussian_filter(image.astype(np.float), sigma=1.)
        return image, {perturbation_name: perturbation_idx}

    return perturbation_fun
