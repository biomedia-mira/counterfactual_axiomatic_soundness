import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, NamedTuple, Optional, Tuple, Union

import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
import numpy as np
import tensorflow as tf
from jax import vmap
from more_itertools import powerset
from numpy.typing import NDArray
from staxplus import Array, KeyArray, Shape
from typing_extensions import Protocol

Image = NDArray[np.uint8]


class ConfoundingFn(Protocol):
    def __call__(self, dataset: tf.data.Dataset, confound: bool = True) -> Tuple[NDArray[Any], Dict[str, NDArray[Any]]]:
        ...


class Oracle(Protocol):
    def __call__(self, image: Image) -> Union[int, float]:
        ...


@dataclass(frozen=True)
class ParentDist:
    name: str
    dim: int
    is_discrete: bool
    is_invertible: bool
    samples: NDArray[Any]
    oracle: Optional[Oracle] = None

    def sample(self, rng: KeyArray, sample_shape: Shape) -> Array:
        sample = random.choice(rng, self.samples, shape=[int(s) for s in sample_shape])
        if self.is_discrete:
            return nn.one_hot(sample, num_classes=self.dim)
        else:
            return sample[..., jnp.newaxis]

    def _oracle(self, image: Array) -> Array:
        if self.oracle is not None:
            return jnp.array(self.oracle(np.array(127.5 * image + 127.5).astype(np.uint8)))
        else:
            raise ValueError('Parent distribution does not have an oracle.')

    def measure(self, image: Array) -> Array:
        return vmap(self._oracle)(image)


class PMF(Protocol):
    def __call__(self, parents: Dict[str, Array]) -> Tuple[Array, Dict[str, NDArray[np.int64]], Dict[str, int]]:
        ...


def get_joint_pmf(histogram: NDArray[np.int64], bin_edges_dict: Dict[str, Union[int, NDArray[np.int64]]]) -> PMF:
    def pmf(parents: Dict[str, Array]) -> Tuple[Array, Dict[str, NDArray[np.int64]], Dict[str, int]]:
        assert parents.keys() == bin_edges_dict.keys()
        indices = []
        for parent_name, bin_edges in bin_edges_dict.items():
            if isinstance(bin_edges, int):
                indices.append(jnp.argmax(parents[parent_name], axis=-1))
            else:
                indices.append(np.digitize(parents[parent_name], (*bin_edges[:-1], bin_edges[-1] + 1))[:, 0] - 1)
        binned_parents = {parent_name: i for parent_name, i in zip(parents.keys(), indices)}
        dims = {parent_name: bin_edges if isinstance(bin_edges, int) else len(bin_edges) - 1
                for parent_name, bin_edges in bin_edges_dict.items()}
        return histogram[tuple(indices)] / np.sum(histogram), binned_parents, dims
    return pmf


class Scenario(NamedTuple):
    train_data: Dict[FrozenSet[str], tf.data.Dataset]
    test_data: tf.data.Dataset
    parent_dists: Dict[str, ParentDist]
    input_shape: Shape
    joint_pmf: PMF


def make_histogram(parents: Dict[str, NDArray[Any]],
                   parent_dists: Dict[str, ParentDist],
                   num_bins: int = 10) -> Tuple[NDArray[np.int64], NDArray[np.int64], PMF]:
    indicator = {}
    bin_edges_dict = {}
    for (parent_name, parent_dist), parent in zip(parent_dists.items(), parents.values()):
        if parent_dist.is_discrete:
            _parent, dim = parent, parent_dist.dim
            bin_edges_dict[parent_name] = dim
        else:
            _, bin_edges = np.histogram(parents[parent_name], bins=num_bins)
            _parent, dim = np.digitize(parent, (*bin_edges[:-1], bin_edges[-1] + 1)) - 1, num_bins
            bin_edges_dict[parent_name] = bin_edges
        indicator[parent_name] = np.array([_parent == i for i in range(dim)])
    shape = [parent_dist.dim if parent_dist.is_discrete else num_bins for parent_dist in parent_dists.values()]
    index_map = np.array([np.logical_and.reduce(a) for a in itertools.product(*indicator.values())])
    index_map = index_map.reshape((*shape, -1))
    histogram = np.sum(index_map, axis=-1)
    index_map = np.argwhere(np.moveaxis(index_map, -1, 0))[..., 1:]

    if np.any(histogram == 0):
        message = '\n'.join([', '.join([f'{parent} == {val[i]}' for i, parent in enumerate(parents)])
                            for val in np.argwhere(histogram == 0)])
        warnings.warn(f'Distribution does not have full support in:\n{message}')

    return histogram, index_map, get_joint_pmf(histogram, bin_edges_dict)


def get_simulated_intervention_datasets(dataset: tf.data.Dataset,
                                        parents: Dict[str, NDArray[Any]],
                                        parent_dists: Dict[str, ParentDist],
                                        num_bins: int = 10) \
        -> Tuple[Dict[FrozenSet[str], tf.data.Dataset], PMF]:
    histogram, index_map, pmf = make_histogram(parents, parent_dists, num_bins=num_bins)
    joint_dist = histogram / np.sum(histogram)

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

        num_repeats_per_element = np.take(num_repeats, np.ravel_multi_index(index_map.T.tolist(), num_repeats.shape)).T
        tf_num_repeats_per_element = tf.convert_to_tensor(num_repeats_per_element)

        def resample_fn(index: tf.Tensor, data: Tuple[tf.Tensor, Dict[str, tf.Tensor]]) -> tf.data.Dataset:
            image, parents = data
            return tf.data.Dataset.from_tensors((image, parents)).repeat(tf_num_repeats_per_element[index])
        unconfounded_dataset = dataset.enumerate().flat_map(resample_fn)
        unconfounded_dataset = unconfounded_dataset.shuffle(buffer_size=np.sum(histogram * num_repeats),
                                                            reshuffle_each_iteration=True)
        datasets[frozenset(parent_set)] = unconfounded_dataset
    return datasets, pmf
