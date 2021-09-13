import itertools
from typing import Dict, Tuple, Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from more_itertools import powerset


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


def get_resample_fn(num_repeats: tf.Tensor, parent_dims: Dict[str, int]) -> Callable[
    [tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]], tf.data.Dataset]:
    def resample_fn(index: tf.Tensor, image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> tf.data.Dataset:
        _num_repeats = num_repeats[[parents[key] for key in parent_dims.keys()]]
        return tf.data.Dataset.from_tensors((index, image, parents)).repeat(_num_repeats)

    return resample_fn


def get_marginal_datasets(dataset: tf.data.Dataset, parents: Dict[str, np.ndarray], parent_dims: Dict[str, int]) \
        -> Tuple[tf.data.Dataset, Dict[str, np.ndarray]]:
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

    return tf.data.Dataset.zip(datasets), marginals


def get_unconfounded_datasets(dataset: tf.data.Dataset, parents: Dict[str, np.ndarray], parent_dims: Dict[str, int],
                              batch_size: int, img_encode_fn) -> Tuple[tf.data.Dataset, Dict[str, np.ndarray]]:
    dataset, marginals = get_marginal_datasets(dataset, parents, parent_dims)

    def encode(*args):
        _, image, parents = args
        parents = {parent: tf.one_hot(value, parent_dims[parent]) for parent, value in parents.items()}
        return img_encode_fn(image), parents

    dataset = dataset.map(lambda d: {key: encode(*value) for key, value in d.items()})
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return dataset, marginals
