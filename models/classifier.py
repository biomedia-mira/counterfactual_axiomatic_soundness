from typing import Any, Callable, Dict, Sequence, Tuple

import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.example_libraries import stax
from jax.example_libraries.optimizers import Optimizer, OptimizerState
from jax.example_libraries.stax import Dense, Flatten, LogSoftmax

from components import Array, KeyArray, Model, Params, StaxLayer, UpdateFn


def classifier(num_classes: int, layers: Sequence[StaxLayer]) -> Model:
    init_fn, classify_fn = stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax)

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Tuple[Array, Dict[str, Array]]:
        image, target = inputs
        prediction = classify_fn(params, image)
        accuracy = jnp.equal(jnp.argmax(prediction, axis=-1), jnp.argmax(target, axis=-1))
        cross_entropy = -jnp.sum(prediction * target, axis=-1)
        return jnp.mean(cross_entropy), {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    def init_optimizer_fn(params: Params, optimizer: Optimizer) \
            -> Tuple[OptimizerState, UpdateFn, Callable[[OptimizerState], Callable]]:
        opt_init, opt_update, get_params = optimizer

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs)
            opt_state = opt_update(i, grads, opt_state)
            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    return init_fn, apply_fn, init_optimizer_fn
