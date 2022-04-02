from typing import Any, Callable, Dict, Sequence, Tuple

import jax.numpy as jnp
from jax import jit, random, tree_map, value_and_grad
from jax.example_libraries.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.example_libraries.stax import Dense, FanInConcat, LeakyRelu, parallel, serial

from components import Array, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.f_gan import f_gan
from components.standard_vae import vae
from components.stax_extension import Pass
from models.functional_counterfactual import MechanismFn
from models.utils import concat_parents


def vae_gan(parent_dims: Dict[str, int],
            latent_dim: int,
            critic_layers: Sequence[StaxLayer],
            encoder_layers: Sequence[StaxLayer],
            decoder_layers: Sequence[StaxLayer],
            from_joint: bool = True) -> Tuple[Model, Callable[[Params], MechanismFn]]:
    assert len(parent_dims) > 0
    source_dist = frozenset() if from_joint else frozenset(parent_dims.keys())
    target_dist = source_dist
    hidden_dim = 256
    encoder = serial(parallel(serial(*encoder_layers), Pass), FanInConcat(axis=-1),
                     Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu)
    decoder = serial(FanInConcat(axis=-1), *decoder_layers)
    vae_init_fn, vae_apply_fn = vae(latent_dim, encoder, decoder, conditional=True, bernoulli_ll=True)
    gan_init_fn, gan_apply_fn = f_gan(critic=serial(*critic_layers), mode='gan', trick_g=True)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Params:
        c_shape = (-1, sum(parent_dims.values()))
        _, gan_params = gan_init_fn(rng, (input_shape, c_shape))
        output_shape, vae_params = vae_init_fn(rng, (input_shape, c_shape))
        return output_shape, (gan_params, vae_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Any]:
        gan_params, vae_params = params
        (image, parents) = inputs[source_dist]
        vae_loss, recon, vae_output = vae_apply_fn(vae_params, rng, (image, concat_parents(parents)))
        p_sample = (inputs[target_dist][0], concat_parents(inputs[target_dist][1]))
        q_sample = (recon, concat_parents(parents))
        gan_loss, gan_output = gan_apply_fn(gan_params, p_sample, q_sample)
        loss = vae_loss  # + gan_loss

        # conditional samples just for visualisation
        sample_parents = {name: jnp.eye(dim)[random.randint(rng, (image.shape[0],), 0, dim)]
                          for name, dim in parent_dims.items()}
        _, samples, _ = vae_apply_fn(vae_params, rng, (image, concat_parents(sample_parents)))
        output = {'image': image, 'samples': samples, 'loss': loss[jnp.newaxis], **vae_output, **gan_output}
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
            return vae_apply_fn(params[1], rng, (image, concat_parents(do_parents)))[1]

        return mechanism_fn

    return (init_fn, apply_fn, init_optimizer_fn), get_mechanism_fn
