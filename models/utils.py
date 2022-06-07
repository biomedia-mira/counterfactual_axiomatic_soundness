from typing import Dict, FrozenSet, Tuple

import jax.numpy as jnp
from staxplus import Array, ArrayTree, KeyArray
from typing_extensions import Protocol, TypeGuard


class AuxiliaryFn(Protocol):
    def __call__(self, image: Array, parent: Array) -> Tuple[Array, Dict[str, Array]]:
        ...


class MechanismFn(Protocol):
    def __call__(self,
                 rng: KeyArray,
                 image: Array,
                 parents: Dict[str, Array],
                 do_parents: Dict[str, Array]) -> Array:
        ...


def concat_parents(parents: Dict[str, Array]) -> Array:
    return jnp.concatenate([parents[parent_name] for parent_name in sorted(parents.keys())], axis=-1)


def is_inputs(inputs: ArrayTree) -> TypeGuard[Dict[FrozenSet[str], Tuple[Array, Dict[str, Array]]]]:
    return isinstance(inputs, dict) \
        and all([isinstance(k1, frozenset)
                 and isinstance(v1, tuple)
                 and isinstance(v1[0], Array)
                 and isinstance(v1[1], dict)
                 and all([isinstance(k2, str) and isinstance(v2, Array) for k2, v2 in v1[1].items()])
                 for k1, v1 in inputs.items()])
