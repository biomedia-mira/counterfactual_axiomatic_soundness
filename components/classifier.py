from typing import Any, Dict, Iterable, Tuple

import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten, LogSoftmax

from components.stax_extension import Array, Params, StaxLayer


def classifier(num_classes: int, layers: Iterable[StaxLayer]) -> StaxLayer:
    init_fn, classify_fn = stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax)

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Tuple[Array, Dict[str, Array]]:
        image, target = inputs
        prediction = classify_fn(params, image)
        accuracy = jnp.equal(jnp.argmax(prediction, axis=-1), jnp.argmax(target, axis=-1))
        cross_entropy = -jnp.sum(prediction * target, axis=-1)
        return jnp.mean(cross_entropy), {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    return init_fn, apply_fn
