import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Tuple

import numpy as np
import tensorflow as tf
from more_itertools import powerset
from numpy.typing import NDArray
from tqdm import tqdm

IMAGE = NDArray[np.uint8]
ConfoundingFn = Callable[[IMAGE, int], Tuple[IMAGE, int]]


@dataclass
class Distribution:
    dim: int
    marginal: NDArray
    continuous: bool = False


def image_gallery(array: np.ndarray, ncols: int = 16, num_images_to_display: int = 128) -> np.ndarray:
    array = np.clip(array, a_min=0, a_max=255) / 255.
    array = array[::len(array) // num_images_to_display]
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols + int(bool(nindex % ncols))
    pad = np.zeros(shape=(nrows * ncols - nindex, height, width, intensity))
    array = np.concatenate((array, pad), axis=0)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def get_uniform_confusion_matrix(num_rows: int, num_columns: int) -> NDArray[np.float_]:
    return np.ones((num_rows, num_columns)) / num_columns


def get_random_confusion_matrix(num_rows: int, num_columns: int, temperature: float = .1, seed: int = 1) \
        -> NDArray[np.float_]:
    random_state = np.random.RandomState(seed=seed)
    logits = random_state.random(size=(num_rows, num_columns))
    tmp = np.exp(logits / temperature)
    return np.array(tmp / tmp.sum(1, keepdims=True))


def get_diagonal_confusion_matrix(num_rows: int, num_columns: int, noise: float = 0.) -> NDArray[np.float_]:
    assert num_rows == num_columns
    return np.array((np.eye(num_rows) * (1. - noise)) + (np.ones((num_rows, num_rows))
                                                         - np.eye(num_rows)) * noise / (num_rows - 1))


def apply_confounding_fns_to_dataset(dataset: tf.data.Dataset, confounding_fns: List[ConfoundingFn],
                                     parent_dims: Dict[str, int]) -> Tuple[IMAGE, Dict[str, NDArray[np.int_]]]:
    image_list, parents_list = [], []
    for image, label in tqdm(dataset.as_numpy_iterator()):
        parents = [int(label)]
        for confounding_fn in confounding_fns:
            image, new_parent = confounding_fn(image, label)
            parents.append(new_parent)
        image_list.append(image)
        parents_list.append(np.array(parents).astype(np.int64))
    image_dataset = np.array(image_list)
    parents_dataset = {key: np.array(parents_list)[:, i] for i, key in enumerate(parent_dims.keys())}
    return image_dataset, parents_dataset


def get_resample_fn(num_repeats: tf.Tensor, parent_dims: Dict[str, int]) \
        -> Callable[[tf.Tensor, Dict[str, tf.Tensor]], tf.data.Dataset]:
    def resample_fn(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> tf.data.Dataset:
        _num_repeats = num_repeats[[tf.argmax(parents[key]) for key in parent_dims.keys()]]
        return tf.data.Dataset.from_tensors((image, parents)).repeat(_num_repeats)

    return resample_fn


def get_marginal_datasets(dataset: tf.data.Dataset, parents: Dict[str, np.ndarray], parent_dims: Dict[str, int]) \
        -> Tuple[Dict[FrozenSet, tf.data.Dataset], Dict[str, Distribution]]:
    indicator = {key: [parents[key] == i for i in range(dim)] for key, dim in parent_dims.items()}
    index_map = np.array([np.logical_and.reduce(a) for a in itertools.product(*indicator.values())])
    index_map = index_map.reshape((*parent_dims.values(), -1))
    counts = np.sum(index_map, axis=-1)
    if np.any(counts == 0):
        raise ValueError('Distribution does not have full support.')
    joint_dist = counts / np.sum(counts)
    datasets, marginals = {}, {}

    for parent_set in powerset(parents.keys()):
        axes = tuple(np.flatnonzero(np.array([parent in parent_set for parent in parents])))
        marginal_dist = np.ones(shape=(1,) * len(parents))
        for axis in axes:
            marginal_dist = marginal_dist * \
                            np.sum(joint_dist, axis=tuple(set(range(counts.ndim)) - {axis}), keepdims=True)

        dist = marginal_dist * np.sum(joint_dist, axis=axes, keepdims=True)
        weights = dist / counts
        num_repeats = tf.convert_to_tensor(np.round(weights / np.min(weights)).astype(int))
        unconfounded_dataset = dataset.flat_map(get_resample_fn(num_repeats, parent_dims))
        unconfounded_dataset = unconfounded_dataset.shuffle(buffer_size=np.sum(counts * num_repeats),
                                                            reshuffle_each_iteration=True)
        datasets[frozenset(parent_set)] = unconfounded_dataset
        if len(parent_set) == 1:
            parent_name = parent_set[0]
            marginals[parent_name] = Distribution(parent_dims[parent_name], np.squeeze(marginal_dist))

    return datasets, marginals


def load_cached_dataset(dataset_dir: Path, dataset: tf.data.Dataset, confounding_fns: List[ConfoundingFn],
                        parent_dims: Dict[str, int]) -> Tuple[tf.data.Dataset, Dict[str, NDArray]]:
    parents_path = str(dataset_dir / 'parents.npy')
    images_path = str(dataset_dir / 'images.npy')
    try:
        images = np.load(images_path)
        parents = np.load(parents_path, allow_pickle=True).item()
    except FileNotFoundError:
        print('Dataset not found, creating new copy...')
        images, parents = apply_confounding_fns_to_dataset(dataset, confounding_fns, parent_dims)
        dataset_dir.mkdir(exist_ok=True, parents=True)
        np.save(images_path, images)
        np.save(parents_path, parents)
    dataset = tf.data.Dataset.from_tensor_slices((images, parents))
    return dataset, parents
