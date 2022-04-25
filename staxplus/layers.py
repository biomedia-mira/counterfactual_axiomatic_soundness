from functools import partial
from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap
from jax.example_libraries.stax import Conv, LeakyRelu, serial
from jax.nn import normalize
from jax.nn.initializers import ones, zeros

from staxplus.types import (Array, ArrayTree, KeyArray, Params, Shape, ShapeTree, StaxInitialiazer, StaxLayer,
                            is_array_sequence, is_shape, is_shape_sequence)


def stax_wrapper(fn: Callable[[ArrayTree], ArrayTree]) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        return input_shape, ()

    def apply_fn(params: Params, inputs: ArrayTree, **kwargs: Any) -> ArrayTree:
        return fn(inputs)

    return StaxLayer(init_fn, apply_fn)


def layer_norm(axis: Union[int, Tuple[int, ...]],
               bias_init: StaxInitialiazer = zeros,
               scale_init: StaxInitialiazer = ones) -> StaxLayer:
    _axis = axis if isinstance(axis, tuple) else tuple((axis,))

    def init_fun(rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        assert is_shape(input_shape)
        features_shape = tuple(s if i in _axis else 1 for i, s in enumerate(input_shape))
        bias = bias_init(rng, features_shape)
        scale = scale_init(rng, features_shape)
        return input_shape, (bias, scale)

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        bias, scale = params
        return scale * normalize(inputs, axis=axis) + bias

    return StaxLayer(init_fun, apply_fun)


def reshape(output_shape: Shape) -> StaxLayer:

    def init_fun(rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        assert is_shape(input_shape)
        return output_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return jnp.reshape(inputs, output_shape)

    return StaxLayer(init_fun, apply_fun)


def resize(output_shape: Shape, method: str = 'nearest') -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        assert is_shape(input_shape)
        return output_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return vmap(partial(jax.image.resize, shape=output_shape[1:], method=method))(inputs)

    return StaxLayer(init_fn, apply_fn)


def broadcast_together(axis: int = -1) -> StaxLayer:
    def broadcast(array: Array, shape: Shape) -> Array:
        return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        assert is_shape_sequence(input_shape)
        # and all([is_shape(el) for el in input_shape])
        ax = axis % len(input_shape[0])
        out_shape = tuple((*input_shape[0][:ax], shape[axis], *input_shape[0][ax + 1:]) for shape in input_shape)
        return out_shape, ()

    def apply_fn(params: Params, inputs: ArrayTree, **kwargs: Any) -> ArrayTree:
        assert is_array_sequence(inputs)
        ax = axis % len(inputs[0].shape)
        out_shape = inputs[0].shape
        broadcasted = [broadcast(arr, (*out_shape[:ax], arr.shape[axis], *out_shape[ax + 1:])) for arr in inputs[1:]]
        return (inputs[0], *broadcasted)

    return StaxLayer(init_fn, apply_fn)


def ResBlock(out_features: int, filter_shape: Tuple[int, int], strides: Tuple[int, int]) -> StaxLayer:
    PixelNorm2D = layer_norm(axis=(3,))
    _init_fn, _apply_fn = serial(Conv(out_features, filter_shape=(3, 3), strides=(1, 1), padding='SAME'),
                                 PixelNorm2D, LeakyRelu,
                                 Conv(out_features, filter_shape=filter_shape, strides=strides, padding='SAME'),
                                 PixelNorm2D, LeakyRelu)

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        output = _apply_fn(params, inputs)
        if output.shape[-1] < inputs.shape[-1]:
            _residual = inputs[..., :output.shape[-1]]
        else:
            _residual = jnp.repeat(inputs, output.shape[-1] // inputs.shape[-1], axis=-1)
        output = output + jax.image.resize(_residual, shape=output.shape, method='nearest')
        return output

    return StaxLayer(_init_fn, apply_fn)
