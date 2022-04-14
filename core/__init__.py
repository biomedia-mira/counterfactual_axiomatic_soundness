from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple, TypeVar, Union

import jax.numpy as jnp
from jax.random import KeyArray
from optax import GradientTransformation, OptState, Params

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
UpdateFn = Callable[[Params, GradientTransformation, OptState, Any, KeyArray], Tuple[Params, OptState, Array, Any]]
#InitOptimizerFn = Callable[[Opt]]

Model = Tuple[InitFn, ApplyFn, UpdateFn]
