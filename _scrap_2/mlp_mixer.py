from typing import Tuple

import jax.numpy as jnp
from jax import jit
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

    @jit
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



# import jax
# def a():
#
#     layers = (Flatten, Dense(1))
#     channel_init_fun, channel_decoder_apply_fun = stax.serial(*layers)
#
#
#     def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
#         pass
#
#     def apply_fun(params: List[Tuple[jnp.ndarray, ...]],
#                   dense_dct_seq: jnp.ndarray,
#                   parents: Dict[str, jnp.ndarray],
#                   do_parent: jnp.ndarray):
#
#         def body_fun(i, val):
#             val.at[:, i, 0] = channel_decoder_apply_fun(params, val)
#             c = f()
#             p = g()
#             v = z()
#
#         val = jax.lax.fori_loop(1, 10, )
#
#     return init_fun, apply_fun
#
#
# def seq_decoder(seq_len):
#     val = jnp.ones((128, 500, 4))*-1
#     val = jax.lax.fori_loop(1, 10, )
#
#     def fori_loop(lower, upper, body_fun, init_val):
#         val = init_val
#         for i in range(lower, upper):
#             val = body_fun(i, val)
#         return val
