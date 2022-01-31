from typing import Any, Callable, Dict, FrozenSet, Tuple

import jax.numpy as jnp
from jax import jit, random, tree_map, value_and_grad, vmap
from jax.experimental.optimizers import Optimizer, OptimizerState, ParamsFn

from components import Array, InitFn, KeyArray, Model, Params, Shape, UpdateFn
from datasets.utils import Distribution

# [[[image, parents]], [score, output]]
ClassifierFn = Callable[[Tuple[Array, Array]], Tuple[Array, Any]]
# [[params, [image, parents], [image, parents]], [div_loss, output]]
DivergenceFn = Callable[[Params, Tuple[Array, Array], Tuple[Array, Array]], Tuple[Array, Any]]
# [[params, image, parent, do_parent, do_noise], do_image]
MechanismFn = Callable[[Params, Array, Array, Array], Array]
# [[rng, batch_size], [do_parent, order]]
SamplingFn = Callable[[KeyArray, int], Tuple[Array, Array]]


def sampling_fn(parent_dist: Distribution, rng: KeyArray, batch_size: int) -> Tuple[Array, Array]:
    parent_dim = parent_dist.dim
    do_parent = random.choice(rng, parent_dim, shape=(batch_size,), p=parent_dist.marginal)
    return jnp.eye(parent_dim)[do_parent], jnp.argsort(do_parent)


def l2(x: Array) -> Array:
    return vmap(lambda arr: jnp.linalg.norm(jnp.ravel(arr), ord=2))(x)


def functional_counterfactual(source_dist: FrozenSet[str],
                              do_parent_name: str,
                              parent_dist: Distribution,
                              classifiers: Dict[str, ClassifierFn],
                              f_divergence: Tuple[InitFn, DivergenceFn],
                              mechanism: Tuple[InitFn, MechanismFn],
                              optimizer: Optimizer,
                              invertible: bool = False) -> Model:
    target_dist = source_dist.union((do_parent_name,))
    f_div_init_fn, f_div_apply_fn = f_divergence
    mechanisms_init_fn, mechanism_apply_fn = mechanism

    def init_fn(rng: KeyArray, input_shape: Shape) -> Params:
        f_div_output_shape, f_div_params = f_div_init_fn(rng, input_shape)
        mechanism_output_shape, mechanism_params = mechanisms_init_fn(rng, input_shape)
        return mechanism_output_shape, (f_div_params, mechanism_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Any]:
        f_div_params, mechanism_params = params
        (image, parents) = inputs[source_dist]
        parent = parents[do_parent_name]

        # sample new parent
        do_parent, order = sampling_fn(parent_dist, rng, image.shape[0])
        do_parents = {**parents, do_parent_name: do_parent}

        # functional counterfactual
        do_image = mechanism_apply_fn(mechanism_params, image, parent, do_parent)
        loss, output = jnp.zeros(()), {'image': image[order], 'do_image': do_image[order]}

        # effectiveness constraint
        for _parent_name, _parent in do_parents.items():
            cross_entropy, output[_parent_name] = classifiers[_parent_name]((do_image, _parent))
            loss = loss + cross_entropy
        f_div_loss, f_div_output = f_div_apply_fn(f_div_params, inputs[target_dist], (do_image, do_parents))
        loss = loss + f_div_loss
        output.update(f_div_output)

        # composition constraint
        image_same = mechanism_apply_fn(mechanism_params, image, parent, parent)
        id_constraint = l2(image - image_same)
        loss = loss + id_constraint
        output.update({'image_same': image_same[order], 'id_constraint': id_constraint, })

        # reversibility constraint
        if invertible:
            image_cycle = mechanism_apply_fn(mechanism_params, do_image, do_parent, parent)
            cycle_constraint = l2(image - image_cycle)
            loss = loss + cycle_constraint
            output.update({'image_cycle': image_cycle, 'cycle_constraint': cycle_constraint})

        return loss, {f'do_{do_parent_name}': {'loss': loss[jnp.newaxis], **output}}

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
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

    return init_fn, apply_fn, init_optimizer_fn
