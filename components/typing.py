from typing import Any, Callable, Tuple, Union
from jax.random import KeyArray
import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import Params

Array = Union[jnp.ndarray, np.ndarray, Any]
Shape = Tuple[int, ...]
PRNGKey = KeyArray
InitFn = Callable[[PRNGKey, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
