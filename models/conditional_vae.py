from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import jax.random as random
import optax
from datasets.utils import ParentDist
from jax import value_and_grad
from jax.tree_util import tree_map
from staxplus import (Array, ArrayTree, GradientTransformation, KeyArray, Model, OptState, Params, ShapeTree, StaxLayer,
                      c_vae)
from staxplus.types import is_shape

from models.utils import AuxiliaryFn, CouterfactualFn, concat_parents, is_inputs


def conditional_vae(do_parent_name: str,
                    parent_dists: Dict[str, ParentDist],
                    pseudo_oracles: Dict[str, AuxiliaryFn],
                    vae_encoder: StaxLayer,
                    vae_decoder: StaxLayer,
                    bernoulli_ll: bool = True,
                    normal_ll_variance: float = 1.,
                    beta: float = 1.,
                    from_joint: bool = False) -> Tuple[Model, Callable[[Params], CouterfactualFn]]:
    assert do_parent_name == 'all'
    parent_dims = tree_map(lambda x: x.dim, parent_dists)
    assert len(parent_dims) > 0
    source_dist = frozenset() if from_joint else frozenset(parent_dists.keys())
    _init_fn, _apply_fn = c_vae(vae_encoder,
                                vae_decoder,
                                input_range=(-1., 1.),
                                beta=beta,
                                bernoulli_ll=bernoulli_ll,
                                normal_ll_variance=normal_ll_variance)

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Params:
        assert is_shape(input_shape)
        c_shape = (-1, sum(parent_dims.values()))
        _, params = _init_fn(rng, (input_shape, c_shape))
        return params

    def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, Dict[str, Array]]:
        assert is_inputs(inputs)
        k1, k2, k3 = random.split(rng, 3)
        image, parents = inputs[source_dist]
        _parents = concat_parents(parents)
        loss, _, vae_output = _apply_fn(params, (image, _parents, _parents), k1)

        # conditional samples just for visualisation
        do_parents = {p_name: p_dist.sample(_k, (image.shape[0], ))
                      for _k, (p_name, p_dist) in zip(random.split(k2, len(parents)), parent_dists.items())}

        _do_parents = concat_parents(do_parents)
        _, samples, _ = _apply_fn(params, (image, _parents, _do_parents), k3)
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
            _, do_image, _ = _apply_fn(params, (image, concat_parents(parents), concat_parents(do_parents)), rng)
            return do_image
        return counterfactual_fn
    return Model(init_fn, apply_fn, update_fn), get_counterfactual_fn
