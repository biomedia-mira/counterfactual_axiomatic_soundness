from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import optax
from staxplus import (Array, ArrayTree, GradientTransformation, KeyArray, Model,
                  OptState, Params, Shape, StaxLayer)
from staxplus.f_gan import f_gan
from jax import value_and_grad, vmap
from jax.tree_util import tree_map

from models.utils import (DiscriminativeFn, MechanismFn, ParentDist,
                          concat_parents, sample_through_shuffling)


def l2(x: Array) -> Array:
    return vmap(lambda arr: jnp.linalg.norm(jnp.ravel(arr), ord=2))(x)


# If do_parent_name =='all' uses full mechanism else uses partial mechanism
def functional_counterfactual(do_parent_name: str,
                              parent_dists: Dict[str, ParentDist],
                              classifiers: Dict[str, DiscriminativeFn],
                              critic: StaxLayer,
                              mechanism: StaxLayer,
                              constraint_function_power: int = 1,
                              from_joint: bool = True) -> Tuple[Model, Callable[[Params], MechanismFn]]:
    """
    Behaves like a partial mechanism if the set of do_parent_names is smaller than parent_dims.keys()
    """
    parent_dims = tree_map(lambda x: x.dim, parent_dists)
    assert len(parent_dims) > 0
    assert do_parent_name in ['all', *parent_dists.keys()]
    assert parent_dims.keys() == classifiers.keys()
    assert constraint_function_power >= 1
    do_parent_names = tuple(parent_dims.keys()) if do_parent_name == 'all' else (do_parent_name,)
    source_dist = frozenset() if from_joint else frozenset(parent_dims.keys())
    target_dist = frozenset(do_parent_names) if from_joint else frozenset(parent_dims.keys())
    divergence_init_fn, divergence_apply_fn = f_gan(critic=critic, mode='gan', trick_g=True)
    mechanism_init_fn, mechanism_apply_fn = mechanism
    _is_invertible = all([parent_dists[parent_name].is_invertible for parent_name in do_parent_names])

    def init_fn(rng: KeyArray, input_shape: Shape) -> Params:
        c_shape = (-1, sum(parent_dims.values()))
        f_div_output_shape, f_div_params = divergence_init_fn(rng, (input_shape, c_shape))
        c_shape = (-1, sum([dim for p_name, dim in parent_dims.items() if p_name in do_parent_names]))
        mechanism_output_shape, mechanism_params = mechanism_init_fn(rng, (input_shape, c_shape, c_shape))
        return f_div_params, mechanism_params

    def parents_to_array(parents: Dict[str, Array]) -> Array:
        return concat_parents({p_name: array for p_name, array in parents.items() if p_name in do_parent_names})

    def apply_mechanism(params: Params, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
        return mechanism_apply_fn(params, (image, parents_to_array(parents), parents_to_array(do_parents)))

    def apply_fn(params: Params, rng: KeyArray, inputs: ArrayTree) -> Tuple[Array, ArrayTree]:
        divergence_params, mechanism_params = params
        (image, parents) = inputs[source_dist]

        # sample new parent(s) and perform functional counterfactual
        do_parents = sample_through_shuffling(rng, parents)
        order = ... if do_parent_name == 'all' else jnp.argsort(jnp.argmax(do_parents[do_parent_name], axis=-1))
        do_image = apply_mechanism(mechanism_params, image, parents, do_parents)

        # effectiveness constraint
        p_sample = (inputs[target_dist][0], concat_parents(inputs[target_dist][1]))
        q_sample = (do_image, concat_parents(do_parents))
        loss, output = divergence_apply_fn(divergence_params, p_sample, q_sample)
        for parent_name, classifier in classifiers.items():
            cross_entropy, output[parent_name] = classifier((do_image, do_parents[parent_name]))
            loss = loss + cross_entropy
        output.update({'image': image[order], 'do_image': do_image[order]})

        # composition constraint
        image_null_intervention = image
        for i in range(1, constraint_function_power + 1):
            image_null_intervention = apply_mechanism(mechanism_params, image_null_intervention, parents, parents)
            composition_constraint = jnp.mean(l2(image - image_null_intervention))
            loss = loss + composition_constraint
            output.update({f'image_null_intervention_{i:d}': image_null_intervention[order],
                           f'composition_constraint_{i:d}': composition_constraint[jnp.newaxis]})

        # reversibility constraint
        if _is_invertible:
            image_cycle: Optional[Array] = None
            for i in range(1, constraint_function_power + 1):
                if image_cycle is None:
                    image_forward = do_image
                else:
                    image_forward = apply_mechanism(mechanism_params, image_cycle, parents, do_parents)
                image_cycle = apply_mechanism(mechanism_params, image_forward, do_parents, parents)
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
        zero_grads = tree_map(lambda x: jnp.zeros_like(x), params)
        # step discriminator
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, inputs, rng)
        updates, opt_state = optimizer.update(updates=(grads[0], zero_grads[1]), state=opt_state, params=params)
        params = optax.apply_updates(params, updates)
        # step generator
        (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(params, inputs, rng)
        updates, opt_state = optimizer.update(updates=(zero_grads[0], grads[1]), state=opt_state, params=params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss, outputs

    def get_mechanism_fn(params: Params) -> MechanismFn:
        def mechanism_fn(rng: KeyArray, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
            return apply_mechanism(params[1], image, parents, do_parents)

        return mechanism_fn

    return Model(init_fn, apply_fn, update_fn), get_mechanism_fn
