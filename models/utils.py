from typing import Dict, Tuple, Optional, Any

import jax.numpy as jnp

from components import Array, Shape, KeyArray, StaxLayer, Params


def concat_parents(parents: Dict[str, Array]) -> Array:
    return jnp.concatenate([parents[parent_name] for parent_name in sorted(parents.keys())], axis=-1)


def rescale(x: Array, x_range: Tuple[float, float], target_range: Tuple[float, float]) -> Array:
    return (x - x_range[0]) / (x_range[1] - x_range[0]) * (target_range[1] - target_range[0]) + target_range[0]
