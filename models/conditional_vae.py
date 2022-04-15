from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.random as random
import optax
from jax import value_and_grad

from core import Array, GradientTransformation, KeyArray, Model, OptState, Params, Shape, StaxLayer
from core.staxplus.conditional_vae import c_vae
from datasets.utils import MarginalDistribution
from models.utils import concat_parents, MechanismFn


def conditional_vae(parent_dims: Dict[str, int],
                    marginal_dists: Dict[str, MarginalDistribution],
                    vae_encoder: StaxLayer,
                    vae_decoder: StaxLayer,
                    from_joint: bool = True) -> Tuple[Model, Callable[[Params], MechanismFn]]:
    """ Implements VAE with independent normal posterior, standard normal prior and, standard normal likelihood (l2)"""
    assert len(parent_dims) > 0
    source_dist = frozenset() if from_joint else frozenset(parent_dims.keys())
    parent_names = parent_dims.keys()
    _init_fn, _apply_fn = c_vae(vae_encoder, vae_decoder, input_range=(-1., 1.), bernoulli_ll=True)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        c_shape = (-1, sum(parent_dims.values()))
        output_shape, params = _init_fn(rng, (input_shape, c_shape))
        return output_shape, params

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        k1, k2 = random.split(rng, 2)
        image, parents = inputs[source_dist]
        _parents = concat_parents(parents)
        loss, recon, vae_output = _apply_fn(params, (image, _parents, _parents), k1)
        # conditional samples just for visualisation, not part of training
        do_parents = {p_name: marginal_dists[p_name].sample(_rng, (image.shape[0],))
                      for _rng, p_name in zip(random.split(k2, len(parent_names)), parent_names)}
        _do_parents = concat_parents(do_parents)
        _, samples, _ = _apply_fn(params, (image, _parents, _do_parents), k2)
        return loss, {'image': image, 'samples': samples, 'loss': loss[jnp.newaxis], **vae_output}

    def update(params: Params, optimizer: GradientTransformation, opt_state: OptState, inputs: Any, rng: KeyArray) \
            -> Tuple[Params, OptState, Array, Any]:
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, inputs, rng=rng)
        updates, opt_state = optimizer.update(updates=grads, state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, outputs

    def get_mechanism_fn(params: Params) -> MechanismFn:
        def mechanism_fn(rng: KeyArray, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
            return _apply_fn(params, (image, concat_parents(parents), concat_parents(do_parents)), rng)[1]

        return mechanism_fn

    return (init_fn, apply_fn, update), get_mechanism_fn
