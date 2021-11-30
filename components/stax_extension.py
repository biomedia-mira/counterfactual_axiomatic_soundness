from typing import Any, Callable, Dict, Tuple, Union

import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.experimental.optimizers import Optimizer, OptimizerState
from jax.experimental.stax import Dense, Flatten, LogSoftmax, ones, serial, zeros
from jax.nn import normalize

from components import Array, PRNGKey, Params, Shape, StaxLayer, UpdateFn


def stax_wrapper(fn: Callable[[Array], Array]) -> StaxLayer:
    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return fn(inputs)

    return init_fn, apply_fn


def layer_norm(axis: Union[int, Tuple[int, ...]], bias_init=zeros, scale_init=ones) -> StaxLayer:
    axis = axis if isinstance(axis, tuple) else tuple((axis,))

    def init_fun(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        features_shape = tuple(s if i in axis else 1 for i, s in enumerate(input_shape))
        bias = bias_init(rng, features_shape)
        scale = scale_init(rng, features_shape)
        return input_shape, (bias, scale)

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        bias, scale = params
        return scale * normalize(inputs, axis=axis) + bias

    return init_fun, apply_fun


def reshape(output_shape: Shape) -> StaxLayer:
    def init_fun(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        return output_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return jnp.reshape(inputs, output_shape)

    return init_fun, apply_fun


# def classifier_head(num_classes: int) -> StaxLayer:
#     init_fn, classify_fn = serial(Flatten, Dense(num_classes), LogSoftmax)
#
#     def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Tuple[Array, Dict[str, Array]]:
#         image, target = inputs
#         prediction = classify_fn(params, image)
#         accuracy = jnp.equal(jnp.argmax(prediction, axis=-1), jnp.argmax(target, axis=-1))
#         cross_entropy = -jnp.sum(prediction * target, axis=-1)
#         return jnp.mean(cross_entropy), {'cross_entropy': cross_entropy, 'accuracy': accuracy}
#
#     return init_fn, apply_fn
#
#
# def optimize(stax_layer: StaxLayer, optimizer: Optimizer):
#     init_fn, apply_fn = stax_layer
#
#     def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn, Callable[[OptimizerState], Callable]]:
#         opt_init, opt_update, get_params = optimizer
#
#         @jit
#         def update(i: int, opt_state: OptimizerState, inputs: Any, rng: PRNGKey) -> Tuple[OptimizerState, Array, Any]:
#             (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs)
#             opt_state = opt_update(i, grads, opt_state)
#             return opt_state, loss, outputs
#
#         return opt_init(params), update, get_params
#
#     return init_fn, apply_fn, init_optimizer_fn
