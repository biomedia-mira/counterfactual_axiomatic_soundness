from typing import Any, Dict, Tuple

import jax.nn as nn
import jax.numpy as jnp
from jax import random, vmap
from jax.example_libraries.stax import Dense, elementwise, FanOut, parallel, serial, Sigmoid, \
    Tanh

from components import Array, KeyArray, Params, Shape, StaxLayer
from models.utils import rescale


@vmap
def calc_kl(mean: Array, scale: Array, eps: float = 1e-12) -> Array:
    variance = scale ** 2.
    return 0.5 * jnp.sum(variance + mean ** 2. - 1. - jnp.log(variance + eps))


@vmap
def calc_bernoulli_log_pdf(image: Array, recon: Array, eps: float = 1e-12) -> Array:
    return jnp.sum(image * jnp.log(recon + eps) + (1. - image) * jnp.log(1. - recon + eps))


@vmap
def calc_normal_log_pdf(image: Array, recon: Array, variance: float = .1) -> Array:
    return -.5 * jnp.sum((image - recon) ** 2. / variance + jnp.log(2 * jnp.pi * variance))


def rsample(rng: KeyArray, mean: Array, scale: Array) -> Array:
    return mean + scale * random.normal(rng, mean.shape)


# numerically stable softplus
def __softplus(x: Array, threshold: float = 20.) -> Array:
    return (nn.softplus(x) * (x < threshold)) + (x * (x >= threshold))


def vae(latent_dim: int, encoder: StaxLayer, decoder: StaxLayer, conditional: bool = False,
        bernoulli_ll: bool = True) -> StaxLayer:
    """
    Standard VAE with standard normal latent prior/posterior and Bernoulli or normal likelihood.
    Expects input to be in the range of -1 to 1
    """
    calc_ll = calc_bernoulli_log_pdf if bernoulli_ll else calc_normal_log_pdf
    enc_init_fn, enc_apply_fn \
        = serial(encoder, FanOut(2), parallel(Dense(latent_dim), serial(Dense(latent_dim)), elementwise(__softplus)))

    dec_init_fn, dec_apply_fn = serial(decoder, Sigmoid if bernoulli_ll else Tanh)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = random.split(rng, 2)
        (enc_output_shape, _), enc_params = enc_init_fn(k1, input_shape)
        dec_input_shape = enc_output_shape if not conditional else (enc_output_shape, *input_shape[1:])
        output_shape, dec_params = dec_init_fn(k2, dec_input_shape)
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, rng: KeyArray, inputs: Any) -> Tuple[Array, Array, Dict[str, Array]]:
        enc_params, dec_params = params
        x = rescale(inputs[0], (-1., 1.), (0., 1.)) if bernoulli_ll else inputs[0]
        c = inputs[1:] if conditional else None
        mean_z, scale_z = enc_apply_fn(enc_params, (x, *c) if conditional else x)
        z = rsample(rng, mean_z, scale_z)
        recon = dec_apply_fn(dec_params, (z, *c) if conditional else z)
        log_px = calc_ll(x, recon)
        kl = calc_kl(mean_z, scale_z)
        elbo = log_px - kl
        loss = jnp.mean(-elbo)
        recon = rescale(recon, (0., 1.), (-1., 1.)) if bernoulli_ll else recon
        avg_mean_z = jnp.mean(mean_z, axis=-1)
        avg_scale_z = jnp.mean(scale_z, axis=-1)
        snr = jnp.abs(avg_mean_z / avg_scale_z)
        return loss, recon, {'recon': recon, 'log_px': log_px, 'kl': kl, 'elbo': elbo,
                             'avg_mean_z': avg_mean_z, 'avg_scale_z': avg_scale_z, 'snr': snr}

    return init_fn, apply_fn
