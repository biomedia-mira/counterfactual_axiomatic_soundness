from typing import Callable, Tuple, Any, Generator, Union, Dict, TypeVar

import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import OptimizerState, Params

T = TypeVar('T')
Tree = Union[Dict[str, T], Dict[str, 'Tree']]
InitFn = Callable[[jnp.ndarray, Tuple[int, ...]], Params]
ApplyFn = Callable[[Params, Tree[np.ndarray]], Tuple[jnp.ndarray, Tree[jnp.ndarray]]]
UpdateFn = Callable[[int, OptimizerState, Tree[np.ndarray]], Tuple[OptimizerState, jnp.ndarray, Tree[jnp.ndarray]]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn]]
DataStream = Callable[..., Generator[Any, None, None]]
