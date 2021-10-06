# https://arxiv.org/abs/1606.00709
from typing import Callable, Iterable, Tuple

import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten
from jax.lax import stop_gradient

from components.typing import Array, Params, StaxLayer

FDivType = Tuple[Callable[[Array], Array], Callable[[Array], Array]]


def gan() -> FDivType:
    def activation(v: Array) -> Array:
        return -jnp.log(1 + jnp.exp(-v))

    def f_conj(t: Array) -> Array:
        return -jnp.log(1 - jnp.exp(t))

    return activation, f_conj


def kl() -> FDivType:
    def activation(v: Array) -> Array:
        return v

    def f_conj(t: Array) -> Array:
        return jnp.exp(t - 1)

    return activation, f_conj


def reverse_kl() -> FDivType:
    def activation(v: Array) -> Array:
        return -jnp.exp(-v)

    def f_conj(t: Array) -> Array:
        return -1 - jnp.log(-t)

    return activation, f_conj


def squared_hellinger() -> FDivType:
    def activation(v: Array) -> Array:
        return 1 - jnp.exp(-v)

    def f_conj(t: Array) -> Array:
        return t / (1 - t)

    return activation, f_conj


def pearson() -> FDivType:
    def activation(v: Array) -> Array:
        return v

    def f_conj(t: Array) -> Array:
        return 1 / 4 * t ** 2 + t

    return activation, f_conj


def jensen_shannon() -> FDivType:
    def activation(v: Array) -> Array:
        return jnp.log(2.) - jnp.log(1 + jnp.exp(-v))

    def f_conj(t: Array) -> Array:
        return -jnp.log(2 - jnp.exp(t))

    return activation, f_conj


def get_activation_and_f_conj(mode: str) -> FDivType:
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


def f_gan(layers: Iterable[StaxLayer], mode: str = 'gan', trick_g: bool = False) -> StaxLayer:
    activation, f_conj = get_activation_and_f_conj(mode)
    init_fun, net_apply_fun = stax.serial(*layers, Flatten, Dense(1))

    def calc_divergence(params: Params, p_sample: Array, q_sample: Array) -> Array:
        t_p_dist = net_apply_fun(params, p_sample)
        t_q_dist = net_apply_fun(params, q_sample)
        return jnp.mean(activation(t_p_dist)) - jnp.mean(f_conj(activation(t_q_dist)))

    def apply_fun(params: Params, p_sample: Array, q_sample: Array) -> Tuple[Array, Array, Array]:
        divergence = calc_divergence(params, p_sample, stop_gradient(q_sample))
        disc_loss = divergence
        if not trick_g:
            gen_loss = calc_divergence(stop_gradient(params), p_sample, q_sample)
        else:
            gen_loss = -jnp.mean(activation(net_apply_fun(stop_gradient(params), q_sample)))
        return divergence, disc_loss, gen_loss

    return init_fun, apply_fun
