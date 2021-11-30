# https://arxiv.org/abs/1606.00709
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax.lax import stop_gradient

from components.stax_extension import Array, Params, StaxLayer

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


def f_gan(critic: StaxLayer, mode: str = 'gan', trick_g: bool = False) -> StaxLayer:
    init_fn, critic_apply_fn = critic
    activation, f_conj = get_activation_and_f_conj(mode)

    def calc_divergence(params: Params, p_sample: Array, q_sample: Array) -> Array:
        t_p_dist = critic_apply_fn(params, p_sample)
        t_q_dist = critic_apply_fn(params, q_sample)
        return jnp.mean(activation(t_p_dist)) - jnp.mean(f_conj(activation(t_q_dist)))

    def apply_fn(params: Params, p_sample: Array, q_sample: Array, **kwargs: Any) -> Tuple[Array, Dict[str, Array]]:
        divergence = calc_divergence(params, p_sample, stop_gradient(q_sample))
        critic_loss = -divergence
        if not trick_g:
            generator_loss = calc_divergence(stop_gradient(params), p_sample, q_sample)
        else:
            generator_loss = -jnp.mean(activation(critic_apply_fn(stop_gradient(params), q_sample)))
        loss = critic_loss + generator_loss
        output = {'divergence': divergence[jnp.newaxis],
                  'critic_loss': critic_loss[jnp.newaxis],
                  'generator_loss': generator_loss[jnp.newaxis]}
        return loss, output

    return init_fn, apply_fn
