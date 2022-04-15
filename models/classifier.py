from typing import Any, Dict, Sequence, Tuple

import jax.numpy as jnp
import optax
from jax import value_and_grad
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Flatten, LogSoftmax

from core import Array, GradientTransformation, KeyArray, Model, OptState, Params, StaxLayer


def classifier(num_classes: int, layers: Sequence[StaxLayer]) -> Model:
    init_fn, classify_fn = stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax)

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Tuple[Array, Dict[str, Array]]:
        image, target = inputs
        prediction = classify_fn(params, image)
        accuracy = jnp.equal(jnp.argmax(prediction, axis=-1), jnp.argmax(target, axis=-1))
        cross_entropy = -jnp.sum(prediction * target, axis=-1)
        return jnp.mean(cross_entropy), {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    def update(params: Params, optimizer: GradientTransformation, opt_state: OptState, inputs: Any, rng: KeyArray) \
            -> Tuple[Params, OptState, Array, Any]:
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, inputs)
        updates, opt_state = optimizer.update(updates=grads, state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, outputs

    return init_fn, apply_fn, update
