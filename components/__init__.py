from typing import Any, Callable, Iterable, Mapping, Tuple, TypeVar, Union, Sequence

import jax.numpy as jnp
from jax.example_libraries.optimizers import Optimizer, OptimizerState, Params, ParamsFn
from jax.random import KeyArray

Array = jnp.ndarray
Shape = Sequence[int]

T = TypeVar('T')
PyTree = Union[T, Iterable['PyTree'], Mapping[Any, 'PyTree']]
ShapeTree = PyTree[Shape]
ArrayTree = PyTree[Array]

InitFn = Callable[[KeyArray, ShapeTree], Tuple[ShapeTree, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
# StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, KeyArray], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params, Optimizer], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]
