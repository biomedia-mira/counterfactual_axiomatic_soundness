from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp

from components import Array, KeyArray

# [[[image, parents]], [score, output]]
ClassifierFn = Callable[[Tuple[Array, Array]], Tuple[Array, Any]]
# [[image, parents, do_parents], do_image]
MechanismFn = Callable[[KeyArray, Array, Dict[str, Array], Dict[str, Array]], Array]


def concat_parents(parents: Dict[str, Array]) -> Array:
    return jnp.concatenate([parents[parent_name] for parent_name in sorted(parents.keys())], axis=-1)
