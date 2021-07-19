# https://arxiv.org/abs/1606.00709
from typing import List, Tuple

import jax.numpy as jnp
from jax import grad, vmap
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten
from jax.lax import stop_gradient
from jax.tree_util import tree_map, tree_reduce


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


def f_gan(mode: str, layers, trick_g: bool = False, gradient_penalty: bool = False):
    activation, f_conj = get_activation_and_f_conj(mode)
    init_fun, net_apply_fun = stax.serial(*layers, Flatten, Dense(1))

    def calc_gradient_penalty(params, p_sample):
        _p = vmap(lambda y: grad(lambda p, x: activation(net_apply_fun(params, x)[0, 0]))(params, y[jnp.newaxis]))(
            p_sample)
        return tree_reduce(lambda x, y: x + y, tree_map(lambda x: jnp.sum(x ** 2.), _p)) / len(p_sample)

    def calc_divergence(params: List[Tuple[jnp.ndarray, ...]], p_sample: jnp.ndarray, q_sample: jnp.ndarray):
        t_p_dist = net_apply_fun(params, p_sample)
        t_q_dist = net_apply_fun(params, q_sample)
        return jnp.mean(activation(t_p_dist)) - jnp.mean(f_conj(activation(t_q_dist)))

    def apply_fun(params: List[Tuple[jnp.ndarray, ...]], p_sample: jnp.ndarray, q_sample: jnp.ndarray):
        divergence = calc_divergence(params, p_sample, stop_gradient(q_sample))
        disc_loss = divergence + (calc_gradient_penalty(params, p_sample) if gradient_penalty else 0.)
        if not trick_g:
            gen_loss = calc_divergence(stop_gradient(params), p_sample, q_sample)
        else:
            gen_loss = -jnp.mean(activation(net_apply_fun(stop_gradient(params), q_sample)))
        return divergence, disc_loss, gen_loss

    return init_fun, apply_fun
