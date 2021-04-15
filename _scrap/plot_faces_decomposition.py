"""
============================
Faces dataset decompositions
============================

This example applies to :ref:`olivetti_faces_dataset` different unsupervised
matrix decomposition (dimension reduction) methods from the module
:py:mod:`sklearn.decomposition` (see the documentation chapter
:ref:`decompositions`) .

"""
print(__doc__)

# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause

import itertools
import logging
from abc import ABC, abstractmethod
from time import time
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.functional
import torchvision
import torchvision.transforms as transforms
from numpy.random import RandomState
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans
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
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


class Colorize(Mechanism):
    def __init__(self, num_colors: int = 10):
        assert num_colors <= 10
        self.color_mapping = torch.zeros(10, 3)
        for label, color in zip(range(10), itertools.cycle(COLORS[:num_colors])):
            self.color_mapping[label] = torch.tensor(color)

    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.cat([image] * 3, dim=0) * self.color_mapping[label].view(3, 1, 1)


class CounfoundedMNIST(Dataset):
    def __init__(self, root: str, mechanisms: Sequence[Mechanism], train: bool = True, download: bool = False) -> None:
        self.base_dataset = torchvision.datasets.MNIST(root=root, train=train, download=download,
                                                       transform=transforms.ToTensor())
        self.mechanisms = mechanisms
        self.train = train

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.base_dataset.__getitem__(index)
        for mechanism in self.mechanisms:
            image = mechanism(image, label if self.train else np.random.randint(0, 10))

        return image, label

    def __len__(self):
        return len(self.base_dataset)


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (3, 28, 28)
rng = RandomState(0)

# #############################################################################
# Load faces data
# faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True,
#                                 random_state=rng)

path = '../data/mnist/'
trainset = torchvision.datasets.MNIST(path, train=True, download=False, transform=transforms.ToTensor())
trainset = CounfoundedMNIST(path, mechanisms=[Colorize(10)], train=True, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)

data = []
for image, _ in trainloader:
    data.append(image.detach().numpy())
data = np.concatenate(data)
faces = data.reshape((data.shape[0], 28 * 28 * 3))
faces = faces[:10000]
n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape).transpose(1, 2, 0), interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


# #############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    ('Eigenfaces - PCA using randomized SVD',
     decomposition.PCA(n_components=n_components, svd_solver='randomized',
                       whiten=True),
     True),

    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
     False),

    ('Independent components - FastICA',
     decomposition.FastICA(n_components=n_components, whiten=True),
     True),

    ('Sparse comp. - MiniBatchSparsePCA',
     decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                      n_iter=100, batch_size=3,
                                      random_state=rng),
     True),

    ('MiniBatchDictionaryLearning',
     decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                               n_iter=50, batch_size=3,
                                               random_state=rng),
     True),

    ('Cluster centers - MiniBatchKMeans',
     MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20,
                     max_iter=50, random_state=rng),
     True),

    ('Factor Analysis components - FA',
     decomposition.FactorAnalysis(n_components=n_components, max_iter=20),
     True),
]

# #############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# #############################################################################
# Do the estimation and plot it

for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces
    if center:
        data = faces_centered
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_

    # Plot an image representing the pixelwise variance provided by the
    # estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
    # via the PCA decomposition, also provides a scalar noise_variance_
    # (the mean of pixelwise variance) that cannot be displayed as an image
    # so we skip it.
    if (hasattr(estimator, 'noise_variance_') and
            estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
        plot_gallery("Pixelwise variance",
                     estimator.noise_variance_.reshape(1, -1), n_col=1,
                     n_row=1)
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

plt.show()

# #############################################################################
# Various positivity constraints applied to dictionary learning.
estimators = [
    ('Dictionary learning',
     decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                               n_iter=50, batch_size=3,
                                               random_state=rng),
     True),
    ('Dictionary learning - positive dictionary',
     decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                               n_iter=50, batch_size=3,
                                               random_state=rng,
                                               positive_dict=True),
     True),
    ('Dictionary learning - positive code',
     decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                               n_iter=50, batch_size=3,
                                               fit_algorithm='cd',
                                               random_state=rng,
                                               positive_code=True),
     True),
    ('Dictionary learning - positive dictionary & code',
     decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                               n_iter=50, batch_size=3,
                                               fit_algorithm='cd',
                                               random_state=rng,
                                               positive_dict=True,
                                               positive_code=True),
     True),
]

# #############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components],
             cmap=plt.cm.RdBu)

# #############################################################################
# Do the estimation and plot it

for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces
    if center:
        data = faces_centered
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    components_ = estimator.components_
    plot_gallery(name, components_[:n_components], cmap=plt.cm.RdBu)

plt.show()
