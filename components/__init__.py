from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import OptimizerState, Params, ParamsFn
from jax.random import KeyArray

from components.stax_extension import layer_norm, reshape

Array = Union[jnp.ndarray, np.ndarray, Any]
Shape = Tuple[int, ...]
PRNGKey = KeyArray
InitFn = Callable[[PRNGKey, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, PRNGKey], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]

Reshape = reshape
LayerNorm2D = layer_norm(axis=(1, 2, 3))
LayerNorm1D = layer_norm(axis=(1,))
PixelNorm2D = layer_norm(axis=(3,))
