from typing import Iterable, Tuple, cast

import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten, LogSoftmax

from components.typing import ApplyFn, InitFn, StaxLayer


def classifier(num_classes: int, layers: Iterable[StaxLayer]) -> Tuple[InitFn, ApplyFn]:
    return stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax)


def calc_accuracy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return cast(jnp.ndarray, jnp.equal(jnp.argmax(pred, axis=-1), jnp.argmax(target, axis=-1)))


def calc_cross_entropy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return cast(jnp.ndarray, -jnp.mean(jnp.sum(pred * target, axis=-1)))
