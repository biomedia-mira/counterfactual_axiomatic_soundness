import itertools
import warnings
from typing import Any, Callable, Dict, FrozenSet, Sequence, Tuple

import numpy as np
import tensorflow as tf
from models.utils import ParentDist
from more_itertools import powerset
from numpy.typing import NDArray
from staxplus import Shape
from typing_extensions import Protocol

IMAGE = NDArray[np.uint8]
Array = NDArray[Any]

Scenario = Tuple[
    Dict[FrozenSet[str], tf.data.Dataset],
    tf.data.Dataset,
    Dict[str, ParentDist],
    Shape]


class ConfoundingFn(Protocol):
    def __call__(self, dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        ...


def _get_histogram(parents: Dict[str, Array],
                   parent_dists: Dict[str, ParentDist],
                   num_bins: int = 10) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    indicator = {}
    for (parent_name, parent_dist), parent in zip(parent_dists.items(), parents.values()):
        if parent_dist.is_discrete:
            _parent, dim = parent, parent_dist.dim
        else:
            _, bin_edges = np.histogram(parents[parent_name], bins=num_bins)
            _parent, dim = np.digitize(parent, (*bin_edges[:-1], bin_edges[-1] + 1)) - 1, num_bins
        indicator[parent_name] = np.array([_parent == i for i in range(dim)])
    shape = [parent_dist.dim if parent_dist.is_discrete else num_bins for parent_dist in parent_dists.values()]
    index_map = np.array([np.logical_and.reduce(a) for a in itertools.product(*indicator.values())])
    index_map = index_map.reshape((*shape, -1))
    histogram = np.sum(index_map, axis=-1)
    index_map = np.argwhere(np.moveaxis(index_map, -1, 0))[..., 1:]
    return histogram, index_map


def get_simulated_intervention_datasets(dataset: tf.data.Dataset,
                                        parents: Dict[str, Array],
                                        parent_dists: Dict[str, ParentDist],
                                        num_bins: int = 10) -> Dict[FrozenSet[str], tf.data.Dataset]:
    histogram, index_map = _get_histogram(parents, parent_dists, num_bins=num_bins)
    joint_dist = histogram / np.sum(histogram)
    if np.any(histogram == 0):
        message = '\n'.join([', '.join([f'{parent} == {val[i]}' for i, parent in enumerate(parents)])
                             for val in np.argwhere(histogram == 0)])
        warnings.warn(f'Distribution does not have full support in:\n{message}')

    datasets = {}
    for parent_set in powerset(parents.keys()):
        axes = tuple(np.flatnonzero(np.array([parent in parent_set for parent in parents])))
        product_of_marginals = np.ones(shape=(1,) * len(parents))
        for axis in axes:
            product_of_marginals = product_of_marginals \
                * np.sum(joint_dist, axis=tuple(set(range(joint_dist.ndim)) - {axis}), keepdims=True)
        interventional_dist = product_of_marginals * np.sum(joint_dist, axis=axes, keepdims=True)
        weights = interventional_dist / histogram
        num_repeats = np.round(weights / np.min(weights[histogram > 0])).astype(int)
        num_repeats[histogram == 0] = 0
        print(f'{str(parent_set)}: max_num_repeat={np.max(num_repeats):d}; total_num_repeats:{np.sum(num_repeats):d}')

        num_repeats_per_element = np.take(num_repeats, np.ravel_multi_index(index_map.T, num_repeats.shape)).T
        tf_num_repeats_per_element = tf.convert_to_tensor(num_repeats_per_element)

        def resample_fn(index: tf.Tensor, data: Tuple[tf.Tensor, Dict[str, tf.Tensor]]) -> tf.data.Dataset:
            image, parents = data
            return tf.data.Dataset.from_tensors((image, parents)).repeat(tf_num_repeats_per_element[index])
        unconfounded_dataset = dataset.enumerate().flat_map(resample_fn)
        unconfounded_dataset = unconfounded_dataset.shuffle(buffer_size=np.sum(histogram * num_repeats),
                                                            reshuffle_each_iteration=True)
        datasets[frozenset(parent_set)] = unconfounded_dataset
    return datasets
