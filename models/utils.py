from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.random as random

from core import Array, KeyArray

# [[[image, parents]], [score, output]]
ClassifierFn = Callable[[Tuple[Array, Array]], Tuple[Array, Any]]
# [[image, parents, do_parents], do_image]
MechanismFn = Callable[[KeyArray, Array, Dict[str, Array], Dict[str, Array]], Array]


def concat_parents(parents: Dict[str, Array]) -> Array:
    return jnp.concatenate([parents[parent_name] for parent_name in sorted(parents.keys())], axis=-1)


def sample_through_shuffling(rng: KeyArray, parents: Dict[str, Array]) -> Dict[str, Array]:
    return {parent_name: random.shuffle(_rng, parent)
            for _rng, (parent_name, parent) in zip(random.split(rng, len(parents)), parents.items())}

