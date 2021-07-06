from typing import Tuple

import jax.lax
import jax.numpy as jnp
import jax.random as random
from jax.nn.initializers import glorot_normal, glorot_uniform


# https://github.com/lucidrains/linformer/blob/55d08cb809472fd0d2b872865f93178d937ed59d/linformer/linformer.py#L66


def Linear(out_dim, W_init=glorot_normal()):
    """Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        return jnp.dot(inputs, params)

    return init_fun, apply_fun


def linear_attention_layer(dim, seq_len, k=20, heads=8, dim_head=None, one_kv_head=False):
    assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

    dim_head = dim_head if dim_head is not None else dim // heads
    kv_dim = dim_head if one_kv_head else (dim_head * heads)
    to_q_init_fn, to_q_apply_fn = Linear(dim_head * heads)
    to_k_init_fn, to_k_apply_fn = Linear(kv_dim)
    to_out_init_fn, to_out_apply_fn = Linear(dim)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        k1, k2, k3, k4 = random.split(rng, 4)
        queries_shape, q_params = to_q_init_fn(k1, input_shape)
        _, k_params = to_k_init_fn(k2, input_shape)
        output_shape, out_params = to_out_init_fn(k3, queries_shape)
        proj_k = glorot_uniform()(k3, (seq_len, k))
        return output_shape, (q_params, k_params, out_params, proj_k)

    def apply_fun(params, x, **kwargs):
        batch_size, n, d = x.shape
        q_params, k_params, out_params, proj_k = params
        assert n == seq_len, f'the sequence length of the key / values must be {seq_len} - {n} given'

        queries = to_q_apply_fn(q_params, x)
        keys = to_k_apply_fn(k_params, x)  # values = keys

        # project keys along the sequence length dimension to k
        keys = jnp.einsum('bnd,nk->bkd', keys, proj_k)

        # merge head into batch for queries and key / values
        queries = jnp.swapaxes(jnp.reshape(queries, (batch_size, n, heads, -1)), 1, 2)
        keys = jnp.swapaxes(jnp.reshape(keys, (batch_size, k, heads, dim_head)), 1, 2)

        # attention
        dots = jnp.einsum('bhnd,bhkd->bhnk', queries, keys) * (dim_head ** -0.5)
        attn = jax.nn.softmax(dots, axis=-1)
        out = jnp.einsum('bhnk,bhkd->bhnd', attn, keys)
        out = jnp.reshape(jnp.swapaxes(out, 1, 2), (batch_size, n, -1))
        return to_out_apply_fn(out_params, out)

    return init_fun, apply_fun
