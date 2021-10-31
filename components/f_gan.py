# https://arxiv.org/abs/1606.00709
from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Dense, Flatten
from jax.lax import stop_gradient
import functools
from jax.tree_util import tree_reduce, tree_map
from components.typing import Array, Params, StaxLayer
from functools import partial
from jax.experimental.maps import xmap


def compose2(f: Callable, g: Callable) -> Callable:
    return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs: Callable) -> Callable:
    return functools.reduce(compose2, fs)


FDivergence = Tuple[Callable[[Array], Array], Callable[[Array], Array]]


def gan() -> FDivergence:
    def activation(v: Array) -> Array:
        return -jnp.log(1 + jnp.exp(-v))

    def f_conj(t: Array) -> Array:
        return -jnp.log(1 - jnp.exp(t))

    return activation, f_conj


def kl() -> FDivergence:
    def activation(v: Array) -> Array:
        return v

    def f_conj(t: Array) -> Array:
        return jnp.exp(t - 1)

    return activation, f_conj


def reverse_kl() -> FDivergence:
    def activation(v: Array) -> Array:
        return -jnp.exp(-v)

    def f_conj(t: Array) -> Array:
        return -1 - jnp.log(-t)

    return activation, f_conj


def squared_hellinger() -> FDivergence:
    def activation(v: Array) -> Array:
        return 1 - jnp.exp(-v)

    def f_conj(t: Array) -> Array:
        return t / (1 - t)

    return activation, f_conj


def pearson() -> FDivergence:
    def activation(v: Array) -> Array:
        return v

    def f_conj(t: Array) -> Array:
        return 1 / 4 * t ** 2 + t

    return activation, f_conj


def jensen_shannon() -> FDivergence:
    def activation(v: Array) -> Array:
        return jnp.log(2.) - jnp.log(1 + jnp.exp(-v))

    def f_conj(t: Array) -> Array:
        return -jnp.log(2 - jnp.exp(t))

    return activation, f_conj


def wasserstein() -> FDivergence:
    def activation(v: Array) -> Array:
        return v

    def f_conj(t: Array) -> Array:
        return t

    return activation, f_conj


def get_activation_and_f_conj(mode: str) -> FDivergence:
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
    elif mode == 'wasserstein':
        return wasserstein()
    else:
        raise ValueError(f'Unsupported divergence: {mode}.')


def f_gan(layers: Iterable[StaxLayer], mode: str = 'gan', trick_g: bool = False, disc_penalty: float = 0.) -> StaxLayer:
    activation, f_conj = get_activation_and_f_conj(mode)
    init_fun, net_apply_fun = stax.serial(*layers)

    def grad_penalty(params: Params, q_sample: Array) -> Array:
        grads_ = jax.jacrev(net_apply_fun)(params, q_sample)

        def tree_norm(grads):
            return tree_reduce(lambda x, y: x + y, tree_map(compose(jnp.sqrt, jnp.sum, jnp.square), grads))

        return jnp.mean(jnp.square(jax.vmap(tree_norm)(grads_) - 1))

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
        loss = gen_loss - disc_loss
        if disc_penalty != 0:
            loss = loss + disc_penalty * grad_penalty(params, stop_gradient(q_sample))
        return loss, disc_loss, gen_loss

    return init_fun, apply_fun
