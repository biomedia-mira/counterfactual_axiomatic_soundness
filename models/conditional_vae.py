from typing import Callable, Dict, Sequence, Tuple

import jax.numpy as jnp
import jax.random as random
import optax
from datasets.utils import ParentDist
from jax import value_and_grad
from staxplus import (Array, ArrayTree, GradientTransformation, KeyArray, Model, OptState, Params, ShapeTree, StaxLayer,
                      c_vae)
from staxplus.types import is_shape

from models.utils import AuxiliaryFn, CouterfactualFn, concat_parents, is_inputs


def conditional_vae(do_parent_names: Sequence[str],
                    parent_dists: Dict[str, ParentDist],
                    pseudo_oracles: Dict[str, AuxiliaryFn],
                    vae_encoder: StaxLayer,
                    vae_decoder: StaxLayer,
                    bernoulli_ll: bool = True,
                    normal_ll_variance: float = 1.,
                    beta: float = 1.,
                    simulated_intervention: bool = True) -> Tuple[Model, Callable[[Params], CouterfactualFn]]:
    """Implements a counterfactual function as conditional VAE.

    Args:
        do_parent_names (Tuple[str, ...]): The set of parents which are intervened upon, the remaining parents are
        unobserved at inference time.
        parent_dists (Dict[str, ParentDist]): The parent distributions.
        pseudo_oracles (Dict[str, AuxiliaryFn]): The pseudo oracles.
        vae_encoder (StaxLayer): The VAE encoder.
        vae_decoder (StaxLayer): The VAE decoder.
        bernoulli_ll (bool, optional): Whether to use a Bernoulli or Normal log-likelihood. Defaults to True.
        normal_ll_variance (float, optional): The fixed value for the variance when using a normal log-likelihood.
        Defaults to 1.
        beta (float, optional): The VAE beta penalty. Defaults to 1..
        simulated_intervention (bool, optional): If True, a simulated intervention making all parents independent is
        used. Otherwise, the model is trained on the original joint distribution. Defaults to True.

    Returns:
        Tuple[Model, Callable[[Params], CouterfactualFn]]: The Model and counterfactual function generator.
    """
    assert all([parent_name in parent_dists.keys() for parent_name in do_parent_names])
    source_dist = frozenset(parent_dists.keys()) if simulated_intervention else frozenset()
    vae_init_fn, vae_apply_fn = c_vae(vae_encoder,
                                      vae_decoder,
                                      input_range=(-1., 1.),
                                      beta=beta,
                                      bernoulli_ll=bernoulli_ll,
                                      normal_ll_variance=normal_ll_variance)

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Params:
        assert is_shape(input_shape)
        c_shape = (-1, sum([dist.dim for p_name, dist in parent_dists.items() if p_name in do_parent_names]))
        _, params = vae_init_fn(rng, (input_shape, c_shape))
        return params

    def parents_to_array(parents: Dict[str, Array]) -> Array:
        return concat_parents({p_name: array for p_name, array in parents.items() if p_name in do_parent_names})

    def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, Dict[str, Array]]:
        assert is_inputs(inputs)
        k1, k2, k3 = random.split(rng, 3)
        image, parents = inputs[source_dist]
        loss, _, vae_output = vae_apply_fn(params, (image, parents_to_array(parents), parents_to_array(parents)), k1)

        # conditional samples just for visualisation
        do_parents = {p_name: p_dist.sample(_k, (image.shape[0], ))
                      for _k, (p_name, p_dist) in zip(random.split(k2, len(parents)), parent_dists.items())}

        _, samples, _ = vae_apply_fn(params, (image, parents_to_array(parents), parents_to_array(do_parents)), k3)
        oracle_output = {p_name: oracle(samples, do_parents[p_name])[1] for p_name, oracle in pseudo_oracles.items()}

        return loss, {'image': image, 'samples': samples, 'loss': loss[jnp.newaxis], **vae_output, **oracle_output}

    def update_fn(params: Params,
                  optimizer: GradientTransformation,
                  opt_state: OptState,
                  rng: KeyArray,
                  inputs: ArrayTree) -> Tuple[Params, OptState, Array, Dict[str, Array]]:
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, rng, inputs)
        updates, opt_state = optimizer.update(updates=grads, state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, outputs

    def get_counterfactual_fn(params: Params) -> CouterfactualFn:
        def counterfactual_fn(rng: KeyArray,
                              image: Array,
                              parents: Dict[str, Array],
                              do_parents: Dict[str, Array]) -> Array:
            _, do_image, _ = vae_apply_fn(params, (image, parents_to_array(parents), parents_to_array(do_parents)), rng)
            return do_image
        return counterfactual_fn
    return Model(init_fn, apply_fn, update_fn), get_counterfactual_fn
