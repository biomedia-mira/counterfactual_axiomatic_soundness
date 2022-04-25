from functools import partial
from typing import Any, Tuple

import jax.nn as nn
import jax.numpy as jnp
from jax import random, vmap

from staxplus.layers import StaxLayer
from staxplus.types import Array, ArrayTree, KeyArray, Params, ShapeTree, is_shape_sequence


def rescale(x: Array, x_range: Tuple[float, float], target_range: Tuple[float, float]) -> Array:
    return (x - x_range[0]) / (x_range[1] - x_range[0]) * (target_range[1] - target_range[0]) + target_range[0]


@vmap
def calc_kl(mean: Array, scale: Array, eps: float = 1e-12) -> Array:
    variance = scale ** 2.
    return 0.5 * jnp.sum(variance + mean ** 2. - 1. - jnp.log(variance + eps))


def rsample(rng: KeyArray, mean: Array, scale: Array) -> Array:
    return mean + scale * random.normal(rng, mean.shape)


@vmap
def calc_bernoulli_log_pdf(image: Array, recon: Array, eps: float = 1e-12) -> Array:
    return jnp.sum(image * jnp.log(recon + eps) + (1. - image) * jnp.log(1. - recon + eps))


@vmap
def calc_normal_log_pdf(image: Array, recon: Array, variance: float = .1) -> Array:
    return -.5 * jnp.sum((image - recon) ** 2. / variance + jnp.log(2 * jnp.pi * variance))


def c_vae(encoder: StaxLayer,
          decoder: StaxLayer,
          input_range: Tuple[float, float] = (0., 1.),
          beta: float = 1.,
          bernoulli_ll: bool = True) -> StaxLayer:
    """ Standard VAE with standard normal latent prior/posterior and Bernoulli or normal likelihood. """
    calc_ll = calc_bernoulli_log_pdf if bernoulli_ll else calc_normal_log_pdf
    enc_init_fn, enc_apply_fn = encoder
    dec_init_fn, dec_apply_fn = decoder

    do_rescale = bernoulli_ll and input_range != (0., 1.)

    def __pass(x: Array) -> Array:
        return x
    _rescale = partial(rescale, x_range=input_range, target_range=(0., 1.)) if do_rescale else __pass
    _undo_rescale = partial(rescale, x_range=(0., 1.), target_range=input_range) if do_rescale else __pass

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Tuple[ShapeTree, Params]:
        assert is_shape_sequence(input_shape)
        k1, k2 = random.split(rng, 2)
        (enc_output_shape, _), enc_params = enc_init_fn(k1, input_shape)
        dec_input_shape = (enc_output_shape, input_shape[1])
        output_shape, dec_params = dec_init_fn(k2, dec_input_shape)
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray, **kwargs: Any) -> ArrayTree:
        enc_params, dec_params = params
        x, z_c, y_c = inputs
        x = _rescale(x)
        mean_z, _scale_z = enc_apply_fn(enc_params, (x, z_c))
        scale_z = nn.softplus(_scale_z)
        z = rsample(rng, mean_z, scale_z)
        recon = dec_apply_fn(dec_params, (z, y_c))
        recon = nn.sigmoid(recon) if bernoulli_ll else recon
        log_px = calc_ll(x, recon)
        kl = calc_kl(mean_z, scale_z)
        elbo = log_px - beta * kl
        loss = jnp.mean(-elbo)
        avg_mean_z = jnp.mean(mean_z, axis=-1)
        avg_scale_z = jnp.mean(scale_z, axis=-1)
        snr = jnp.abs(avg_mean_z / avg_scale_z)
        recon = _undo_rescale(recon)
        return loss, recon, {'recon': recon, 'log_px': log_px, 'kl': kl, 'elbo': elbo,
                             'avg_mean_z': avg_mean_z, 'avg_scale_z': avg_scale_z, 'snr': snr}

    return StaxLayer(init_fn, apply_fn)
