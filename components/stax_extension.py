from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap
from jax.example_libraries.stax import Conv, LeakyRelu, ones, serial, zeros
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


def resize(output_shape: Shape, method: str = 'nearest') -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return output_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return vmap(partial(jax.image.resize, shape=output_shape[1:], method=method))(inputs)

    return init_fn, apply_fn


def _pass() -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return inputs

    return init_fn, apply_fn


def broadcast_together(axis: int = -1):
    def broadcast(array: Array, shape: Shape) -> Array:
        return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

    def init_fn(rng, input_shape):
        ax = axis % len(input_shape[0])
        out_shape = tuple((*input_shape[0][:ax], shape[axis], *input_shape[0][ax + 1:]) for shape in input_shape)
        return out_shape, ()

    def apply_fn(params, inputs, **kwargs):
        ax = axis % len(inputs[0].shape)
        out_shape = inputs[0].shape
        broadcasted = [broadcast(arr, (*out_shape[:ax], arr.shape[axis], *out_shape[ax + 1:])) for arr in inputs[1:]]
        return (inputs[0], *broadcasted)

    return init_fn, apply_fn


def ResBlock(out_features: int, filter_shape: Tuple[int, int], strides: Tuple[int, int]) -> StaxLayer:
    _init_fn, _apply_fn = serial(Conv(out_features, filter_shape=(3, 3), strides=(1, 1), padding='SAME'),
                                 PixelNorm2D, LeakyRelu,
                                 Conv(out_features, filter_shape=filter_shape, strides=strides, padding='SAME'),
                                 PixelNorm2D, LeakyRelu)

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        output = _apply_fn(params, inputs)
        residual = jax.image.resize(jnp.repeat(inputs, output.shape[-1] // inputs.shape[-1], axis=-1),
                                    shape=output.shape, method='nearest')
        return output + residual

    return _init_fn, apply_fn


Reshape = reshape
Resize = resize
Pass = _pass()
BroadcastTogether = broadcast_together
LayerNorm2D = layer_norm(axis=(1, 2, 3))
LayerNorm1D = layer_norm(axis=(1,))
PixelNorm2D = layer_norm(axis=(3,))
