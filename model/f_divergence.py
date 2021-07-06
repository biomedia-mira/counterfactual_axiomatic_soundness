from typing import List, Tuple

import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten


# https://arxiv.org/abs/1606.00709


def gan():
    def activation(v):
        return -jnp.log(1 + jnp.exp(-v))

    def f_conj(t):
        return -jnp.log(1 - jnp.exp(t))

    return activation, f_conj


def kl():
    def activation(v):
        return v

    def f_conj(t):
        return jnp.exp(t - 1)

    return activation, f_conj


def reverse_kl():
    def activation(v):
        return -jnp.exp(-v)

    def f_conj(t):
        return -1 - jnp.log(-t)

    return activation, f_conj


def squared_hellinger():
    def activation(v):
        return 1 - jnp.exp(-v)

    def f_conj(t):
        return t / (1 - t)

    return activation, f_conj


def pearson():
    def activation(v):
        return v

    def f_conj(t):
        return 1 / 4 * t ** 2 + t

    return activation, f_conj


def jensen_shannon():
    def activation(v):
        return jnp.log(2.) - jnp.log(1 + jnp.exp(-v))

    def f_conj(t):
        return -jnp.log(2 - jnp.exp(t))

    return activation, f_conj


def get_activation_and_f_conj(mode: str):
    if mode == 'gan':
        return gan()
    elif mode == 'kl':
        return kl()
    elif mode == 'reverse_kl':
        return reverse_kl()
    elif mode == 'squared_hellinger':
        return squared_hellinger()
    elif mode == 'pearson':
        return pearson()
    elif mode == 'jensen_shannon':
        return jensen_shannon()
    else:
        raise ValueError(f'Unsupported divergence: {mode}.')


def f_divergence(mode: str, layers):
    activation, f_conj = get_activation_and_f_conj(mode)
    init_fun, net_apply_fun = stax.serial(*layers, Flatten, Dense(1))

    def apply_fun(params: List[Tuple[jnp.ndarray, ...]], p_sample: jnp.ndarray, q_sample: jnp.ndarray):
        t_p_dist = net_apply_fun(params, p_sample)
        t_q_dist = net_apply_fun(params, q_sample)
        return jnp.mean(activation(t_p_dist)) - jnp.mean(f_conj(activation(t_q_dist)))

    return init_fun, apply_fun
