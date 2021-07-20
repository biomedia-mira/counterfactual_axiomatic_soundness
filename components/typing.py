from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
from jax.experimental.optimizers import Params

Array = Union[jnp.ndarray, Any]
Shape = Tuple[int, ...]
PRNGKey = jnp.ndarray
InitFn = Callable[[Array, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
