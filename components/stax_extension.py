from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
from jax.experimental.stax import ones, zeros
from jax.nn import normalize

from components import Array, KeyArray, Params, Shape, StaxLayer


def stax_wrapper(fn: Callable[[Array], Array]) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return fn(inputs)

    return init_fn, apply_fn


def layer_norm(axis: Union[int, Tuple[int, ...]], bias_init=zeros, scale_init=ones) -> StaxLayer:
    axis = axis if isinstance(axis, tuple) else tuple((axis,))

    def init_fun(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        features_shape = tuple(s if i in axis else 1 for i, s in enumerate(input_shape))
        bias = bias_init(rng, features_shape)
        scale = scale_init(rng, features_shape)
        return input_shape, (bias, scale)

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        bias, scale = params
        return scale * normalize(inputs, axis=axis) + bias

    return init_fun, apply_fun


def reshape(output_shape: Shape) -> StaxLayer:
    def init_fun(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return output_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return jnp.reshape(inputs, output_shape)

    return init_fun, apply_fun


Reshape = reshape
LayerNorm2D = layer_norm(axis=(1, 2, 3))
LayerNorm1D = layer_norm(axis=(1,))
PixelNorm2D = layer_norm(axis=(3,))
