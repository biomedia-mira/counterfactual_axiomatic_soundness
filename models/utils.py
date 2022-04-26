from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import jax.random as random
from numpy.typing import NDArray
from staxplus import Array, ArrayTree, KeyArray, Shape
from typing_extensions import Protocol
import jax.nn as nn


@dataclass(frozen=True)
class ParentDist:
    name: str
    dim: int
    is_discrete: bool
    is_invertible: bool
    samples: NDArray[Any]

    def sample(self, rng: KeyArray, sample_shape: Shape) -> Array:
        sample = random.choice(rng, self.samples, shape=sample_shape)
        if self.is_discrete:
            return nn.one_hot(sample, num_classes=self.dim)
        else:
            return sample[..., jnp.newaxis]


class DiscriminativeFn(Protocol):
    def __call__(self, image: Array, parent: Array) -> Tuple[Array, ArrayTree]:
        ...


class MechanismFn(Protocol):
    def __call__(self,
                 rng: KeyArray,
                 image: Array,
                 parents: Dict[str, Array],
                 do_parents: Dict[str, Array]) -> Array:
        ...


def concat_parents(parents: Dict[str, Array]) -> Array:
    return jnp.concatenate([parents[parent_name] for parent_name in sorted(parents.keys())], axis=-1)


def sample_through_shuffling(rng: KeyArray, parents: Dict[str, Array]) -> Dict[str, Array]:
    return {parent_name: random.shuffle(_rng, parent)
            for _rng, (parent_name, parent) in zip(random.split(rng, len(parents)), parents.items())}
