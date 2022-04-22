from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterable, Mapping, NamedTuple,
                    NewType, Sequence, Tuple, TypeVar, Union)

import jax.numpy as jnp
from jax._src.random import KeyArray
from jax.numpy import shape
from optax import GradientTransformation, OptState, Params
from typing_extensions import Protocol, runtime_checkable

Array = jnp.ndarray
Shape = Sequence[int]
ShapeTree = Union[Shape, Iterable['ShapeTree'], Mapping[Any, 'ShapeTree']]
ArrayTree = Union[Array, Iterable['Array'], Mapping[Any, 'Array']]


def flatten_nested_dict(nested_dict: Dict, key: Tuple = ()) -> Dict:
    new_dict = {}
    for sub_key, value in nested_dict.items():
        new_key = (*key, sub_key)
        if isinstance(value, dict):
            new_dict.update(flatten_nested_dict(value, new_key))
        else:
            new_dict.update({new_key: value})
    return new_dict


class StaxInitFn(Protocol):
    def __call__(self, rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        ...


class StaxLayer(NamedTuple):
    init: StaxInitFn
    apply: Callable


class InitFn(Protocol):
    def __call__(self, rng: KeyArray, input_shape: ShapeTree) -> Params:
        ...


class ApplyFn(Protocol):
    def __call__(self, params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, ArrayTree]:
        ...


class UpdateFn(Protocol):
    def __call__(self, params: Params,
                 optimizer: GradientTransformation,
                 opt_state: OptState,
                 rng: KeyArray,
                 inputs: ArrayTree) -> Tuple[Params, OptState, Array, ArrayTree]:
        ...


class Model(NamedTuple):
    init: InitFn
    apply: ApplyFn
    update: UpdateFn
