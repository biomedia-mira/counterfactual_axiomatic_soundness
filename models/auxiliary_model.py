from typing import Callable, Tuple, Dict

import jax.numpy as jnp
import optax
from datasets.utils import ParentDist
from jax import value_and_grad
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Flatten
from jax.nn import log_softmax
from staxplus import Array, ArrayTree, GradientTransformation, KeyArray, Model, OptState, Params, ShapeTree, StaxLayer

from models.utils import AuxiliaryFn


def auxiliary_model(parent_dist: ParentDist, backbone: StaxLayer) -> Tuple[Model, Callable[[Params], AuxiliaryFn]]:
    _init_fn, _apply_fn = stax.serial(backbone, Flatten, Dense(parent_dist.dim))

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Params:
        return _init_fn(rng, input_shape)[1]

    def classification(params: Params, image: Array, target: Array) -> Tuple[Array, Dict[str, Array]]:
        prediction = log_softmax(_apply_fn(params, image))
        accuracy = jnp.equal(jnp.argmax(prediction, axis=-1), jnp.argmax(target, axis=-1))
        cross_entropy = -jnp.sum(prediction * target, axis=-1)
        return jnp.mean(cross_entropy), {'cross_entropy': cross_entropy,
                                         'accuracy': accuracy,
                                         'target': target,
                                         'prediction': prediction}

    def regression(params: Params, image: Array, target: Array) -> Tuple[Array, Dict[str, Array]]:
        prediction = _apply_fn(params, image)
        mse = jnp.squeeze(jnp.square(target - prediction))
        absolute_error = jnp.squeeze(jnp.abs(target - prediction))
        relative_error = absolute_error / jnp.squeeze(jnp.abs(target)) * 100.
        return jnp.mean(mse), {'mse': mse,
                               'absolute_error': absolute_error,
                               'relative_error': relative_error,
                               'target': target,
                               'prediction': prediction}

    def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, Dict[str, Array]]:
        image, parent = inputs
        return classification(params, image, parent) if parent_dist.is_discrete else regression(params, image, parent)

    def update_fn(params: Params,
                  optimizer: GradientTransformation,
                  opt_state: OptState,
                  rng: KeyArray,
                  inputs: ArrayTree) -> Tuple[Params, OptState, Array, Dict[str, Array]]:
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, rng, inputs)
        updates, opt_state = optimizer.update(updates=grads, state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, outputs

    def get_auxiliary_fn(params: Params) -> AuxiliaryFn:
        def auxiliary_fn(image: Array, parent: Array) -> Tuple[Array, Dict[str, Array]]:
            return apply_fn(params, None, (image, parent))

        return auxiliary_fn

    return Model(init_fn, apply_fn, update_fn), get_auxiliary_fn
