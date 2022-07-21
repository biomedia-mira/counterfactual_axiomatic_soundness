from typing import Any, Callable, Dict, Optional, Tuple, cast, Sequence

import jax.numpy as jnp
import jax.random as random
import optax
from datasets.utils import ParentDist
from jax import value_and_grad, vmap
from jax.tree_util import tree_map
from staxplus import Array, ArrayTree, GradientTransformation, KeyArray, Model, OptState, Params, ShapeTree, StaxLayer
from staxplus.f_gan import f_gan

from models.utils import AuxiliaryFn, CouterfactualFn, concat_parents, is_inputs


def l2(x: Array) -> Array:
    return vmap(lambda arr: jnp.linalg.norm(jnp.ravel(arr), ord=2))(x)  # type: ignore


def conditional_gan(do_parent_names: Sequence[str],
                    parent_dists: Dict[str, ParentDist],
                    pseudo_oracles: Dict[str, AuxiliaryFn],
                    critic: StaxLayer,
                    generator: StaxLayer,
                    use_composition_constraint: bool = True,
                    use_reversibility_constraint: bool = False,
                    constraint_function_power: int = 1,
                    simulated_intervention: bool = True) -> Tuple[Model, Callable[[Params], CouterfactualFn]]:
    """Implements a counterfactual function as conditional GAN.

    Args:
        do_parent_names (Tuple[str, ...]): The set of parent which are intervened upon, the remaining parents are
        unobserved at inference time.
        parent_dists (Dict[str, ParentDist]): The parent distributions.
        pseudo_oracles (Dict[str, AuxiliaryFn]): The pseudo oracles.
        critic (StaxLayer): The GAN critic. The critic sees all parents during since it is only used for training.
        generator (StaxLayer): The GAN generator which will serve as the counterfactual function.
        use_composition_constraint (bool): Whether to use a composition constraint.
        use_reversibility_constraint (bool): Whether to use a reversibility constraint.
        constraint_function_power (int, optional): The function power for the composition and reversibility constraints.
        Defaults to 1.
        simulated_intervention (bool, optional): If True, a simulated intervention making all parents independent is
        used on the source and target distribution. If False, no simulated intervention is used on the source
        distribution. However, a partial simulated intervention on the target distribution is used to account
        for the fact that the do_parents are independently sampled and thus marginally distributed. Defaults to True.

    Returns:
        Tuple[Model, Callable[[Params], CouterfactualFn]]: The model and counterfactual function generator.
    """

    assert all([parent_name in parent_dists.keys() for parent_name in do_parent_names])
    assert constraint_function_power >= 1
    source_dist = frozenset(parent_dists.keys()) if simulated_intervention else frozenset()
    target_dist = frozenset(parent_dists.keys()) if simulated_intervention else frozenset(do_parent_names)
    divergence_init_fn, divergence_apply_fn = f_gan(critic=critic, mode='gan', trick_g=True)
    generator_init_fn, generator_apply_fn = generator

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Params:
        c_shape = (-1, sum([dist.dim for _, dist in parent_dists.items()]))
        _, f_div_params = divergence_init_fn(rng, (input_shape, c_shape))
        c_shape = (-1, sum([dist.dim for p_name, dist in parent_dists.items() if p_name in do_parent_names]))
        _, generator_params = generator_init_fn(rng, (input_shape, c_shape, c_shape))
        return f_div_params, generator_params

    def parents_to_array(parents: Dict[str, Array]) -> Array:
        return concat_parents({p_name: array for p_name, array in parents.items() if p_name in do_parent_names})

    def counterfactual_fn(params: Params,
                          image: Array,
                          parents: Dict[str, Array],
                          do_parents: Dict[str, Array]) -> Array:
        return cast(Array, generator_apply_fn(params, (image, parents_to_array(parents), parents_to_array(do_parents))))

    def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, ArrayTree]:
        assert(is_inputs(inputs))
        divergence_params, generator_params = params
        (image, parents) = inputs[source_dist]

        # sample new parent(s) and perform functional counterfactual
        do_parents = {p_name: p_dist.sample(_rng, (image.shape[0], )) if p_name in do_parent_names else parents[p_name]
                      for _rng, (p_name, p_dist) in zip(random.split(rng, len(parents)), parent_dists.items())}
        order = jnp.argsort(jnp.argmax(do_parents[do_parent_names[0]], axis=-1))
        do_image = counterfactual_fn(generator_params, image, parents, do_parents)

        output: ArrayTree = {}
        # measure parents (only for logging purposes)
        for parent_name, oracle in pseudo_oracles.items():
            _, output[parent_name] = oracle(do_image, do_parents[parent_name])

        # effectiveness constraint
        p_sample = (inputs[target_dist][0], concat_parents(inputs[target_dist][1]))
        q_sample = (do_image, concat_parents(do_parents))
        loss, div_output = divergence_apply_fn(divergence_params, p_sample, q_sample)
        output.update({'image': image[order], 'do_image': do_image[order], **div_output})

        # composition constraint
        if use_composition_constraint:
            image_null_intervention = image
            for i in range(1, constraint_function_power + 1):
                image_null_intervention = counterfactual_fn(generator_params, image_null_intervention, parents, parents)
                composition_constraint = jnp.mean(l2(image - image_null_intervention))
                loss = loss + composition_constraint
                output.update({f'image_null_intervention_{i:d}': image_null_intervention[order],
                               f'composition_constraint_{i:d}': composition_constraint[jnp.newaxis]})

        # reversibility constraint ()
        if use_reversibility_constraint:
            image_cycle: Optional[Array] = None
            for i in range(1, constraint_function_power + 1):
                if image_cycle is None:
                    image_forward = do_image
                else:
                    image_forward = counterfactual_fn(generator_params, image_cycle, parents, do_parents)
                image_cycle = counterfactual_fn(generator_params, image_forward, do_parents, parents)
                reversibility_constraint = jnp.mean(l2(image - image_cycle))
                loss = loss + reversibility_constraint
                output.update({f'image_cycle_{i:d}': image_cycle,
                               f'reversibility_constraint_{i:d}': reversibility_constraint[jnp.newaxis]})

        return loss, {'loss': loss[jnp.newaxis], **output}

    def update_fn(params: Params,
                  optimizer: GradientTransformation,
                  opt_state: OptState,
                  rng: KeyArray,
                  inputs: Any) -> Tuple[Params, OptState, Array, Any]:
        k1, k2 = random.split(rng, 2)
        zero_grads = tree_map(lambda x: jnp.zeros_like(x), params)
        # step discriminator
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, k1, inputs)
        updates, opt_state = optimizer.update(updates=(grads[0], zero_grads[1]), state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        # step generator
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, k2, inputs)
        updates, opt_state = optimizer.update(updates=(zero_grads[0], grads[1]), state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, outputs

    def get_counterfactual_fn(params: Params) -> CouterfactualFn:
        def _counterfactual_fn(rng: KeyArray,
                               image: Array,
                               parents: Dict[str, Array],
                               do_parents: Dict[str, Array]) -> Array:
            _, generator_params = params
            return counterfactual_fn(generator_params, image, parents, do_parents)

        return _counterfactual_fn

    return Model(init_fn, apply_fn, update_fn), get_counterfactual_fn
