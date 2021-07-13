from typing import Callable, Tuple

import jax.numpy as jnp
from jax.experimental.optimizers import Params

Array = jnp.ndarray
Shape = Tuple[int, ...]
InitFn = Callable[[Array, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Callable[..., Tuple[InitFn, ApplyFn]]
