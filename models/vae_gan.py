from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.random as random
from jax import jit, value_and_grad
from jax.example_libraries.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.tree_util import tree_map

from components import Array, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.conditional_vae import c_vae
from components.f_gan import f_gan
from datasets.utils import MarginalDistribution
from models.utils import concat_parents, MechanismFn


def vae_gan(parent_dims: Dict[str, int],
            marginal_dists: Dict[str, MarginalDistribution],
            vae_encoder: StaxLayer,
            vae_decoder: StaxLayer,
            critic: StaxLayer,
            from_joint: bool = True) -> Tuple[Model, Callable[[Params], MechanismFn]]:
    """ Implements VAE with independent normal posterior, standard normal prior and, standard normal likelihood (l2)"""
    assert len(parent_dims) > 0
    source_dist = frozenset() if from_joint else frozenset(parent_dims.keys())
    target_dist = source_dist
    parent_names = parent_dims.keys()
    vae_init_fn, vae_apply_fn = c_vae(vae_encoder, vae_decoder, input_range=(-1., 1.), bernoulli_ll=True)
    gan_init_fn, gan_apply_fn = f_gan(critic=critic, mode='gan', trick_g=True)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        c_shape = (-1, sum(parent_dims.values()))
        output_shape, vae_params = vae_init_fn(rng, (input_shape, c_shape))
        _, gan_params = gan_init_fn(rng, (output_shape, c_shape))
        return output_shape, (vae_params, gan_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        vae_params, gan_params = params
        k1, k2 = random.split(rng, 2)
        image, parents = inputs[source_dist]
        vae_loss, recon, vae_output = vae_apply_fn(vae_params, (image, concat_parents(parents)), k1)
        sample_parents = {p_name: marginal_dists[p_name].sample(_rng, (image.shape[0],))
                          for _rng, p_name in zip(random.split(k2, len(parent_names)), parent_names)}
        _, samples, _ = vae_apply_fn(vae_params, (image, concat_parents(sample_parents)), k2)

        p_sample = (inputs[target_dist][0], concat_parents(inputs[target_dist][1]))
        q_sample = (recon, concat_parents(parents))
        gan_loss, gan_output = gan_apply_fn(gan_params, p_sample, q_sample)
        loss = vae_loss + gan_loss
        return loss, {'image': image, 'samples': samples, 'loss': loss[jnp.newaxis], **vae_output, **gan_output}

    def init_optimizer_fn(params: Params, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizer

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            zero_grads = tree_map(lambda x: x * 0, grads)
            opt_state = opt_update(i, (grads[0], zero_grads[1]), opt_state)
            for _ in range(1):
                (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
                opt_state = opt_update(i, (zero_grads[0], grads[1]), opt_state)
            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    def get_mechanism_fn(params: Params):
        def mechanism_fn(rng: KeyArray, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
            return vae_apply_fn(params[0], (image, concat_parents(do_parents)), rng)[1]

        return mechanism_fn

    return (init_fn, apply_fn, init_optimizer_fn), get_mechanism_fn
