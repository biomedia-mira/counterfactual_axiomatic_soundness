from typing import Any, Iterable, Mapping, NamedTuple, Sequence, Tuple, Union

import jax.numpy as jnp
from jax._src.random import KeyArray
from optax import GradientTransformation, OptState, Params
from typing_extensions import Protocol, TypeGuard

KeyArray = KeyArray
GradientTransformation = GradientTransformation
OptState = OptState
Params = Params
Array = jnp.ndarray
Shape = Sequence[int]
ShapeTree = Union[Shape, Iterable['ShapeTree'], Mapping[Any, 'ShapeTree']]
ArrayTree = Union[Array, Iterable['Array'], Mapping[Any, 'Array']]


def is_shape(shape: ShapeTree) -> TypeGuard[Shape]:
    return isinstance(shape, Sequence) and all([isinstance(el, int) for el in shape])


def is_shape_sequence(shape: ShapeTree) -> TypeGuard[Sequence[Shape]]:
    return isinstance(shape, Sequence) and all([isinstance(el, Sequence) and is_shape(el) for el in shape])


def is_array_sequence(array: ArrayTree) -> TypeGuard[Sequence[Array]]:
    return all([isinstance(el, Array) for el in array]) and isinstance(array, Sequence)


class StaxInitFn(Protocol):
    def __call__(self, rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        ...


class StaxApplyFn(Protocol):
    def __call__(self, params: Params, *args: Any, **kwargs: Any) -> ArrayTree:
        ...


class StaxLayer(NamedTuple):
    init: StaxInitFn
    apply: StaxApplyFn


DType = Any


class StaxInitialiazer(Protocol):
    def __call__(self, key: KeyArray, shape: Shape, dtype: DType = jnp.float_) -> Array:
        ...


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
