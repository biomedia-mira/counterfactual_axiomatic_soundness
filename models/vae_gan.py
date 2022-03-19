from typing import Any, Callable, Dict, Sequence, Tuple

import jax.numpy as jnp
from jax import jit, random, tree_map, value_and_grad, vmap
from jax.experimental.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.experimental.stax import Dense, Flatten, Softplus, parallel, serial, FanOut, Exp

from components import Array, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.f_gan import f_gan
from components.stax_extension import stax_wrapper
from models.functional_counterfactual import MechanismFn, condition_on_parents


def standard_vae(parent_dims: Dict[str, int],
                 latent_dim: int,
                 encoder_layers: Sequence[StaxLayer],
                 decoder_layers: Sequence[StaxLayer]) -> StaxLayer:
    """ Implements VAE with independent normal posterior, standard normal prior and, standard normal likelihood (l2)"""
    assert len(parent_dims) > 0
    enc_init_fn, enc_apply_fn = serial(*encoder_layers, Flatten, FanOut(2),
                                       parallel(Dense(latent_dim), serial(Dense(latent_dim), Exp)))
    dec_init_fn, dec_apply_fn = serial(*decoder_layers)

    def kl(mu, variance):
        return jnp.mean(0.5 * jnp.sum(variance + mu ** 2. - 1. - jnp.log(variance), axis=-1))

    def rsample(rng, mu, variance):
        return mu + jnp.sqrt(variance) * random.normal(rng, mu.shape)

    def log_pdf(image, recon) -> Array:
        return jnp.mean(vmap(lambda x, y: -jnp.sum((x - y) ** 2.))(image, recon))

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        c_dim = sum(parent_dims.values())
        (enc_output_shape, _), enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + c_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, rng: KeyArray, image: Array, parents: Dict[str, Array]) \
            -> Tuple[Array, Array, Dict[str, Array]]:
        enc_params, dec_params = params
        mean_z, variance_z = enc_apply_fn(enc_params, image)
        z = rsample(rng, mean_z, variance_z)
        _parents = [parents[parent_name] for parent_name in sorted(parent_dims.keys())]
        latent_code = jnp.concatenate((z, *_parents), axis=-1)
        recon = dec_apply_fn(dec_params, latent_code)
        recon_loss = log_pdf(image, recon)
        kl_loss = kl(mean_z, variance_z)
        elbo = recon_loss - kl_loss
        return elbo, recon, {'recon': recon, 'recon_loss': recon_loss, 'kl_loss': kl_loss, 'elbo': elbo}

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
        loss = -elbo  + gan_loss
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
