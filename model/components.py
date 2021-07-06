from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten, Gelu, LogSoftmax, Relu
from model.linear_attention import linear_attention_layer
from model.f_divergence import f_divergence


# differentiable rounding operation
def differentiable_round(x):
    return x - (jax.lax.stop_gradient(x) - jnp.round(x))


def layer_norm(axes: Tuple[int, ...] = (0, 2, 1)):
    _init_fun = lambda rng, input_shape: (input_shape, ())
    _apply_fun = lambda params, x, **kwargs: (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(
        jnp.var(x, axis=-1, keepdims=True) + 1e-5)
    return _init_fun, _apply_fun


# Classifier
def classifier(num_classes: int, seq_shape: Tuple[int, int]):
    seq_len, seq_dim = seq_shape
    layers = (layer_norm(), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Gelu,
              layer_norm(), Dense(seq_dim // 2), Gelu,
              layer_norm(), Dense(seq_dim // 2), Gelu,
              Flatten, Dense(num_classes), LogSoftmax)
    return stax.serial(*layers)


# KL estimator
def f_divergence_estimator(seq_shape: Tuple[int, int]):
    seq_len, seq_dim = seq_shape
    layers = (layer_norm(), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Gelu,
              layer_norm(), Dense(seq_dim // 2), Gelu,
              layer_norm(), linear_attention_layer(seq_dim // 2, seq_len, heads=seq_dim // 2), Gelu,
              layer_norm(), Dense(seq_dim // seq_dim))
    return f_divergence(mode='kl', layers=layers)


# mechanism that acts on image based on categorical parent variable
def mechanism(parent_name: str,
              parent_dims: Dict[str, int],
              seq_shape: Tuple[int, int]):
    seq_len, seq_dim = seq_shape
    input_seq_dim = sum(parent_dims.values()) + parent_dims[parent_name] + seq_dim

    layers = (layer_norm(), Dense(10), Gelu,
              layer_norm(), linear_attention_layer(10, seq_len, heads=10), Gelu,
              layer_norm(), Dense(seq_dim), Gelu,
              layer_norm(), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Gelu,
              layer_norm(), Dense(seq_dim), Gelu,
              layer_norm(), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Gelu, Dense(seq_dim))

    net_init_fun, net_apply_fun = stax.serial(*layers)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        return net_init_fun(rng, (*input_shape[:2], input_seq_dim))

    def apply_fun(params: List[Tuple[jnp.ndarray, ...]],
                  dense_dct_seq: jnp.ndarray,
                  parents: Dict[str, jnp.ndarray],
                  do_parent: jnp.ndarray):
        def broadcast(parent: jnp.ndarray) -> jnp.ndarray:
            return jnp.broadcast_to(jnp.expand_dims(parent, axis=1), (*dense_dct_seq.shape[:2], parent.shape[1]))

        parents_as_seq = broadcast(jnp.concatenate([*parents.values(), do_parent], axis=-1))
        return net_apply_fun(params, jnp.concatenate((dense_dct_seq, parents_as_seq), axis=-1))

    return init_fun, apply_fun
