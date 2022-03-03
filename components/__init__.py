from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import OptimizerState, Params, ParamsFn, Optimizer
from jax.random import KeyArray

Array = Union[jnp.ndarray, np.ndarray, Any]
Shape = Tuple[int, ...]
InitFn = Callable[[KeyArray, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, KeyArray], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params, Optimizer], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]
