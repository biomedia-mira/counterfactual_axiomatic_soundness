from typing import Tuple

import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Gelu


def transpose(axes: Tuple[int, ...] = (0, 2, 1)):
    _init_fun = lambda rng, input_shape: (tuple(input_shape[axis] for axis in axes), ())
    _apply_fun = lambda params, inputs, **kwargs: jnp.transpose(inputs, axes=axes)
    return _init_fun, _apply_fun


def mixer_layer(dim_0: int, dim_1: int):
    mlp1_init_fun, mlp1_apply_fun = stax.serial(*(transpose(), Dense(dim_0), Gelu, Dense(dim_0), transpose()))
    mlp2_init_fun, mlp2_apply_fun = stax.serial(*(Dense(dim_1), Gelu, Dense(dim_1)))

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        output_shape, p1 = mlp1_init_fun(rng, input_shape)
        output_shape, p2 = mlp2_init_fun(rng, output_shape)
        return output_shape, (p1, p2)

    def apply_fun(params, inputs, **kwargs):
        p1, p2 = params
        outputs_mlp1 = mlp1_apply_fun(p1, inputs)
        if inputs.shape == outputs_mlp1.shape:
            outputs_mlp1 = inputs + outputs_mlp1
        outputs_mlp2 = mlp2_apply_fun(p2, outputs_mlp1)
        if outputs_mlp1.shape == outputs_mlp2.shape:
            outputs_mlp2 = outputs_mlp1 + outputs_mlp2
        return outputs_mlp2

    return init_fun, apply_fun
