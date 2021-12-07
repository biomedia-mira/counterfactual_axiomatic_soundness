from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.experimental.optimizers import OptimizerState, Params, ParamsFn
from jax.random import KeyArray

from components.f_gan import f_gan
from components.stax_extension import layer_norm, reshape

Array = Union[jnp.ndarray, np.ndarray, Any]
Shape = Tuple[int, ...]
InitFn = Callable[[KeyArray, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, KeyArray], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]

Reshape = reshape
LayerNorm2D = layer_norm(axis=(1, 2, 3))
LayerNorm1D = layer_norm(axis=(1,))
PixelNorm2D = layer_norm(axis=(3,))

__all__ = ['Array',
           'Params',
           'Shape',
           'KeyArray',
           'InitFn',
           'ApplyFn',
           'StaxLayer',
           'StaxLayerConstructor',
           'UpdateFn',
           'InitOptimizerFn',
           'Model',
           'Reshape',
           'LayerNorm2D',
           'LayerNorm1D',
           'PixelNorm2D',
           'f_gan']
