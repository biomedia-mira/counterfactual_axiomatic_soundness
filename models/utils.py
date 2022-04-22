from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp
import jax.random as random
from numpy.typing import NDArray
from typing_extensions import Protocol

from core import Array, ArrayTree, KeyArray, Shape


@dataclass(frozen=True)
class ParentDist:
    name: str
    dim: int
    is_discrete: bool
    is_invertible: bool
    samples: NDArray

    def sample(self, rng: KeyArray, sample_shape: Shape) -> Array:
        return random.choice(rng, self.samples, shape=sample_shape)


class DiscriminativeFn(Protocol):
    def __call__(self, image: Array, parent: Array) -> Tuple[Array, ArrayTree]:
        pass


class MechanismFn(Protocol):
    def __call__(self,
                 rng: KeyArray,
                 image: Array,
                 parents: Dict[str, Array],
                 do_parents: Dict[str, Array]) -> Array:
        pass


def concat_parents(parents: Dict[str, Array]) -> Array:
    return jnp.concatenate([parents[parent_name] for parent_name in sorted(parents.keys())], axis=-1)


def sample_through_shuffling(rng: KeyArray, parents: Dict[str, Array]) -> Dict[str, Array]:
    return {parent_name: random.shuffle(_rng, parent)
            for _rng, (parent_name, parent) in zip(random.split(rng, len(parents)), parents.items())}
