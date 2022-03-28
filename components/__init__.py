from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
from jax.example_libraries.optimizers import Optimizer, OptimizerState, Params, ParamsFn
from jax.random import KeyArray
from numpy.typing import NDArray

Array = Union[jnp.ndarray, NDArray, Any]
Shape = Tuple[int, ...]
InitFn = Callable[[KeyArray, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, KeyArray], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params, Optimizer], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]
