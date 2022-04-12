from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import colors
from numpy.typing import NDArray
from PIL import Image
from scipy.stats.contingency import crosstab
from tqdm import tqdm

from datasets.utils import get_marginal_datasets
from datasets.utils import Scenario


# Colligation/Yule/Phi coefficient https://en.wikipedia.org/wiki/Coefficient_of_colligation
def colligation_coefficient(var1: NDArray, var2: NDArray) -> float:
    a = np.sum(np.logical_and(var1 == 0, var2 == 0))
    b = np.sum(np.logical_and(var1 == 1, var2 == 0))
    c = np.sum(np.logical_and(var1 == 0, var2 == 1))
    d = np.sum(np.logical_and(var1 == 1, var2 == 1))
    t1, t2 = np.sqrt(a * d), np.sqrt(b * c)
    return float((t1 - t2) / (t1 + t2))


def parwise_colligation_coefficient(vars: NDArray) -> NDArray:
    assert vars.ndim == 2
    nvars = vars.shape[0]
    table = np.zeros(shape=(nvars, nvars))
    for i, j in product(range(nvars), range(nvars)):
        if i > j:
            continue
        cc = colligation_coefficient(vars[i], vars[j])
        table[i, j] = cc
        table[j, i] = cc
    return table


def analyse_dataset(images: NDArray, parents: Dict[str, NDArray]) -> None:
    parent_names = list(parents.keys())
    _parents = np.array(list(parents.values()))
    ct = crosstab(*_parents)[1]
    plt.figure(figsize=(6, 6))
    plt.imshow(ct, plt.get_cmap('Blues'))
    plt.colorbar()
    thresh = ct.max() / 2
    for i, j in product(range(ct.shape[0]), range(ct.shape[1])):
        plt.text(j, i, "{:d}".format(ct[i, j]),
                 horizontalalignment="center",
                 color="black" if ct[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()
    ##
    yule_coefficient = parwise_colligation_coefficient(_parents)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(yule_coefficient), cmap='seismic', norm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1))
    plt.colorbar()
    tick_marks = np.arange(len(parent_names))
    plt.xticks(tick_marks, parent_names, rotation=90)
    plt.yticks(tick_marks, parent_names)
    for i, j in product(range(yule_coefficient.shape[0]), range(yule_coefficient.shape[1])):
        plt.text(j, i, "{:0.2f}".format(yule_coefficient[i, j]), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.show()

    for i, j in zip(*np.unravel_index(np.argsort(-np.abs(yule_coefficient), axis=None), yule_coefficient.shape)):
        if i >= j:
            continue
        print(parent_names[i], parent_names[j], yule_coefficient[i, j])
    return


def make_celeb_a_mask_hq_dataset(raw_data_dir: Path, size: Tuple[int, int]) -> Tuple[NDArray, Dict[str, NDArray]]:
    data = pd.read_csv(raw_data_dir / 'CelebAMask-HQ-attribute-anno.txt', delim_whitespace=True, header=1)
    _parents: List[NDArray] = []
    _images: List[NDArray] = []
    for image_name, attr in tqdm(data.iterrows()):
        image_path = raw_data_dir / 'CelebA-HQ-img' / image_name
        attr[attr == -1] = 0
        with Image.open(image_path) as image:
            _images.append(image.resize(size).__array__())
        _parents.append(attr.values)
    images = np.stack(_images)
    parents = np.stack(_parents)
    parents[parents == -1] = 0
    parents_as_dict = dict(zip(data.keys(), parents.T))
    return images, parents_as_dict


def get_celeb_a_mask_hq_dataset(data_dir: Path, raw_data_dir: Optional[Path] = None) -> Tuple[
    NDArray, Dict[str, NDArray]]:
    data_dir = data_dir / 'celeb_a_mask_hq'
    images_path = str(data_dir / 'images.npy')
    parents_path = str(data_dir / 'parents.npy')
    try:
        images = np.load(images_path)
        parents = np.load(parents_path, allow_pickle=True).item()
    except FileNotFoundError:
        assert raw_data_dir is not None
        images, parents = make_celeb_a_mask_hq_dataset(raw_data_dir, size=(128, 128))
        data_dir.mkdir(exist_ok=True, parents=True)
        np.save(images_path, images)
        np.save(parents_path, parents)
    return images, parents


def get_mustache_goatee_dataset(data_dir: Path, raw_data_dir: Optional[Path] = None) -> tf.data.Dataset:
    images, parents = get_celeb_a_mask_hq_dataset(data_dir, raw_data_dir)
    indices = np.where(parents['Male'])[0]
    images = images[indices]
    parents = {key: value[indices] for key, value in parents.items() if key in ['Goatee', 'Mustache']}
    dataset = tf.data.Dataset.from_tensor_slices((images, parents))

    analyse_dataset(images, parents)


def get_encode_fn(parent_dims: Dict[str, int]) \
        -> Callable[[tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
    def encode_fn(image: tf.Tensor, patents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = (tf.cast(image, tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        patents = {parent: tf.one_hot(value, parent_dims[parent]) for parent, value in patents.items()}
        return image, patents

    return encode_fn


def mustache_goatee_scenario(data_dir: Path, raw_data_dir: Optional[Path] = None) -> Scenario:
    parent_dims = {'goatee': 2, 'mustache': 2}
    is_invertible = {'goatee': True, 'mustache': True}
    images, parents = get_celeb_a_mask_hq_dataset(data_dir, raw_data_dir)
    indices = np.where(parents['Male'])[0]
    parents = {key.lower(): value for key, value in parents.items() if key in ['Goatee', 'Mustache']}
    # train test split
    rng = np.random.RandomState(1)
    rng.shuffle(indices)
    train_indices = indices[:int(.7 * len(indices))]
    test_indices = indices[int(.7 * len(indices)):]
    train_images, train_parents = images[train_indices], jax.tree_map(lambda x: x[train_indices], parents)
    test_images, test_parents = images[test_indices], jax.tree_map(lambda x: x[test_indices], parents)
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_parents))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_parents))

    encode_fn = get_encode_fn(parent_dims)
    train_data = train_data.map(encode_fn)
    test_dataset = test_dataset.map(encode_fn)
    train_datasets, marginals = get_marginal_datasets(train_data, train_parents, parent_dims)
    input_shape = (-1, *test_dataset.element_spec[0].shape)
    return train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape
