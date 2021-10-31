from typing import Iterable, cast

import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten, LogSoftmax

from components.typing import Array, StaxLayer


def classifier(num_classes: int, layers: Iterable[StaxLayer]) -> StaxLayer:
    return cast(StaxLayer, stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax))


def calc_accuracy(pred: Array, target: Array) -> Array:
    return cast(Array, jnp.equal(jnp.argmax(pred, axis=-1), jnp.argmax(target, axis=-1)))


def calc_cross_entropy(pred: Array, target: Array, reduce: bool = False) -> Array:
    return -jnp.mean(jnp.sum(pred * target, axis=-1)) if reduce else -jnp.sum(pred * target, axis=-1)
