from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random, tree_map, value_and_grad, vmap
from jax.example_libraries.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.example_libraries.stax import Conv, Relu, Tanh
from jax.example_libraries.stax import Dense, FanInConcat, FanOut, Flatten, LeakyRelu, parallel, serial, elementwise, \
    Sigmoid
from jax.image import resize

from components import Array, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.f_gan import f_gan
from components.stax_extension import Reshape, stax_wrapper
from models.functional_counterfactual import condition_on_parents, MechanismFn
from jax.tree_util import tree_reduce
import jax.nn as nn


def up_sample(new_shape: Shape) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return new_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return vmap(partial(resize, shape=new_shape[1:], method='nearest'))(inputs)

    return init_fn, apply_fn


UpSample = up_sample


def softplus(x, threshold: float = 20.):
    return (nn.softplus(x) * (x < threshold)) + (x * (x >= threshold)) + 1e-5


def standard_vae(parent_dims: Dict[str, int],
                 latent_dim: int,
                 encoder_layers: Sequence[StaxLayer],
                 decoder_layers: Sequence[StaxLayer]) -> StaxLayer:
    """ Implements VAE with independent normal posterior, standard normal prior and, standard normal likelihood (l2)"""
    assert len(parent_dims) > 0
    hidden_dim = 256

    n_channels = hidden_dim // 4

    encoder_layers = (
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Flatten, Dense(hidden_dim), LeakyRelu)
    decoder_layers = (
        Dense(hidden_dim), LeakyRelu, Dense(n_channels * 7 * 7),
        LeakyRelu,
        Reshape((-1, 7, 7, n_channels)), UpSample((-1, 14, 14, n_channels)),
        Conv(n_channels, filter_shape=(5, 5), strides=(1, 1), padding='SAME'), LeakyRelu,
        UpSample((-1, 28, 28, n_channels)),
        Conv(3, filter_shape=(5, 5), strides=(1, 1), padding='SAME'),
        Conv(3, filter_shape=(1, 1), strides=(1, 1), padding='SAME'), Sigmoid)

    enc_init_fn, enc_apply_fn = serial(parallel(serial(*encoder_layers), stax_wrapper(lambda x: x)),
                                       FanInConcat(axis=-1), Dense(hidden_dim), LeakyRelu,
                                       Dense(hidden_dim), LeakyRelu,
                                       FanOut(2),
                                       parallel(Dense(latent_dim), serial(Dense(latent_dim)),
                                                elementwise(softplus)))
    dec_init_fn, dec_apply_fn = serial(FanInConcat(axis=-1), *decoder_layers)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = random.split(rng, 2)
        c_dim = sum(parent_dims.values())
        (enc_output_shape, _), enc_params = enc_init_fn(k1, (input_shape, (-1, c_dim)))
        output_shape, dec_params = dec_init_fn(k2, (enc_output_shape, (-1, c_dim)))
        return output_shape, (enc_params, dec_params)

    @vmap
    def _kl(mean: Array, scale: Array, eps: float = 1e-12) -> Array:
        variance = scale ** 2.
        return 0.5 * jnp.sum(variance + mean ** 2. - 1. - jnp.log(variance + eps))

    @vmap
    def bernoulli_log_pdf(image: Array, recon: Array, eps: float = 1e-12) -> Array:
        return jnp.sum(image * jnp.log(recon + eps) + (1. - image) * jnp.log(1. - recon + eps))

    @vmap
    def normal_log_pdf(image: Array, recon: Array, variance: float = .1) -> Array:
        return -.5 * jnp.sum((image - recon) ** 2. / variance + jnp.log(2 * jnp.pi * variance))

    def rsample(rng: KeyArray, mean: Array, scale: Array) -> Array:
        return mean + scale * random.normal(rng, mean.shape)

    def apply_fn(params: Params, rng: KeyArray, image: Array, parents: Dict[str, Array]) \
            -> Tuple[Array, Array, Dict[str, Array]]:
        image = (image + 1.) / 2.
        enc_params, dec_params = params
        _parents = jnp.concatenate([parents[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        mean_z, scale_z = enc_apply_fn(enc_params, (image, _parents))
        z = rsample(rng, mean_z, scale_z)
        recon = dec_apply_fn(dec_params, (z, _parents))
        log_pdf = bernoulli_log_pdf(image, recon)
        kl = _kl(mean_z, scale_z)
        elbo = log_pdf - kl
        recon = recon * 2. - 1.
        ##
        # conditional samples
        random_parents = {name: jnp.eye(dim)[random.randint(rng, (image.shape[0],), 0, dim)]
                          for name, dim in parent_dims.items()}
        _parents = jnp.concatenate([random_parents[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        mean_z, scale_z = enc_apply_fn(enc_params, (image, _parents))
        random_recon = dec_apply_fn(dec_params, (rsample(rng, mean_z, scale_z), _parents))
        random_recon = random_recon * 2. - 1.
        return elbo, recon, {'recon': recon, 'log_pdf': log_pdf, 'kl': kl, 'elbo': elbo, 'random_recon': random_recon,
                             'variance': jnp.mean(scale_z**2., axis=-1),
                             'mean': jnp.mean(mean_z, axis=-1)}

    return init_fn, apply_fn


def vae_gan(parent_dims: Dict[str, int],
            latent_dim: int,
            critic_layers: Sequence[StaxLayer],
            encoder_layers: Sequence[StaxLayer],
            decoder_layers: Sequence[StaxLayer],
            condition_divergence_on_parents: bool = True,
            from_joint: bool = True) -> Tuple[Model, Callable[[Params], MechanismFn]]:
    assert len(parent_dims) > 0
    source_dist = frozenset() if from_joint else frozenset(parent_dims.keys())
    target_dist = source_dist
    vae_init_fn, vae_apply_fn = standard_vae(parent_dims, latent_dim, encoder_layers, decoder_layers)
    input_layer = condition_on_parents(parent_dims) if condition_divergence_on_parents else stax_wrapper(lambda x: x[0])
    critic = serial(input_layer, *critic_layers, Flatten, Dense(1))
    gan_init_fn, gan_apply_fn = f_gan(critic, mode='gan', trick_g=True)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Params:
        _, gan_params = gan_init_fn(rng, input_shape)
        output_shape, vae_params = vae_init_fn(rng, input_shape)
        return output_shape, (gan_params, vae_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Any]:
        gan_params, vae_params = params
        (image, parents) = inputs[source_dist]
        elbo, recon, vae_output = vae_apply_fn(vae_params, rng, image, parents)
        gan_loss, gan_output = gan_apply_fn(gan_params, inputs[target_dist], (recon, parents))
        # loss = -elbo + gan_loss
        loss = jnp.mean(-elbo)  # + gan_loss
        output = {'image': image, 'loss': loss[jnp.newaxis], **vae_output, **gan_output}
        return loss, output

    def init_optimizer_fn(params: Params, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizer

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            zero_grads = tree_map(lambda x: x * 0, grads)
            opt_state = opt_update(i, (zero_grads[0], grads[1]), opt_state)
            for _ in range(1):
                (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
                opt_state = opt_update(i, (grads[0], zero_grads[1]), opt_state)
            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    def get_mechanism_fn(params: Params) -> MechanismFn:
        def mechanism_fn(rng: KeyArray, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
            return vae_apply_fn(params[1], rng, image, do_parents)[1]

        return mechanism_fn

    return (init_fn, apply_fn, init_optimizer_fn), get_mechanism_fn
