from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional, Tuple

import jax.numpy as jnp
from jax import jit, random, tree_map, value_and_grad, vmap
from jax.experimental.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.experimental.stax import Dense, Flatten, serial
from numpy.typing import NDArray

from components import Array, InitFn, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.f_gan import f_gan
from components.stax_extension import stax_wrapper

# [[[image, parents]], [score, output]]
ClassifierFn = Callable[[Tuple[Array, Array]], Tuple[Array, Any]]
# [[params, [image, parents], [image, parents]], [div_loss, output]]
DivergenceFn = Callable[[Params, Tuple[Array, Array], Tuple[Array, Array]], Tuple[Array, Any]]
# [[params, image, parent, do_parent], do_image]
MechanismFn = Callable[[Params, Array, Array, Array], Array]
# [[rng, batch_size], Tuple[parent_sample, order]]
SamplingFn = Callable[[KeyArray, Shape], Tuple[Array, Array]]


def l2(x: Array) -> Array:
    return vmap(lambda arr: jnp.linalg.norm(jnp.ravel(arr), ord=2))(x)


def get_sampling_fn(dim: int, is_continuous: bool, marginal_dist: NDArray) -> SamplingFn:
    if is_continuous:
        raise NotImplementedError

    def sampling_fn(rng: KeyArray, sample_shape: Shape) -> Tuple[Array, Array]:
        do_parent = random.choice(rng, dim, shape=sample_shape, p=marginal_dist)
        return jnp.eye(dim)[do_parent], jnp.argsort(do_parent)

    return sampling_fn


def condition_on_parents(parent_dims: Dict[str, int]) -> StaxLayer:
    def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
        return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

    def init_fn(rng: KeyArray, shape: Shape) -> Tuple[Shape, Params]:
        return (*shape[:-1], shape[-1] + sum(parent_dims.values())), ()

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Array:
        image, parents = inputs
        shape = (*image.shape[:-1], sum(parent_dims.values()))
        _parents = jnp.concatenate([parents[key] for key in parent_dims.keys()], axis=-1)
        return jnp.concatenate((image, broadcast(_parents, shape)), axis=-1)

    return init_fn, apply_fn


def partial_mechanism(source_dist: FrozenSet[str],
                      parent_dims: Dict[str, int],
                      do_parent_name: str,
                      classifiers: Dict[str, ClassifierFn],
                      critic_layers: Iterable[StaxLayer],
                      mechanism: Tuple[InitFn, MechanismFn],
                      sampling_fn: SamplingFn,
                      is_invertible: bool,
                      optimizer: Optimizer,
                      condition_divergence_on_parents: bool = True,
                      constraint_function_exponent: int = 1) -> Model:
    assert constraint_function_exponent >= 1
    assert len(classifiers) == len(parent_dims) if len(classifiers) > 0 else True
    target_dist = source_dist.union((do_parent_name,))
    input_layer = condition_on_parents(parent_dims) if condition_divergence_on_parents else stax_wrapper(lambda x: x[0])
    divergence_layers = serial(input_layer, *critic_layers, Flatten, Dense(1))
    divergence_init_fn, divergence_apply_fn = f_gan(divergence_layers, mode='gan', trick_g=True)
    mechanism_init_fn, mechanism_apply_fn = mechanism

    def init_fn(rng: KeyArray, input_shape: Shape) -> Params:
        f_div_output_shape, f_div_params = divergence_init_fn(rng, input_shape)
        mechanism_output_shape, mechanism_params = mechanism_init_fn(rng, input_shape)
        return mechanism_output_shape, (f_div_params, mechanism_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Any]:
        divergence_params, mechanism_params = params
        (image, parents) = inputs[source_dist]
        parent = parents[do_parent_name]

        # sample new parent and perform functional counterfactual
        do_parent, order = sampling_fn(rng, (image.shape[0],))
        do_parents = {**parents, do_parent_name: do_parent}
        do_image = mechanism_apply_fn(mechanism_params, image, parent, do_parent)

        # effectiveness constraint
        loss, output = divergence_apply_fn(divergence_params, inputs[target_dist], (do_image, do_parents))
        for parent_name, classifier in classifiers.items():
            cross_entropy, output[parent_name] = classifier((do_image, do_parents[parent_name]))
            loss = loss + cross_entropy
        output.update({'image': image[order], 'do_image': do_image[order]})

        # composition constraint
        image_null_intervention = image
        for i in range(1, constraint_function_exponent + 1):
            image_null_intervention = mechanism_apply_fn(mechanism_params, image_null_intervention, parent, parent)
            composition_constraint = jnp.mean(l2(image - image_null_intervention))
            loss = loss + composition_constraint
            output.update({f'image_null_intervention_{i:d}': image_null_intervention[order],
                           f'composition_constraint_{i:d}': composition_constraint})

        # reversibility constraint
        if is_invertible:
            image_cycle: Optional[Array] = None
            for i in range(1, constraint_function_exponent + 1):
                if image_cycle is None:
                    image_forward = do_image
                else:
                    image_forward = mechanism_apply_fn(mechanism_params, image_cycle, parent, do_parent)
                image_cycle = mechanism_apply_fn(mechanism_params, image_forward, do_parent, parent)
                reversibility_constraint = jnp.mean(l2(image - image_cycle))
                loss = loss + reversibility_constraint
                output.update({f'image_cycle_{i:d}': image_cycle,
                               f'reversibility_constraint_{i:d}': reversibility_constraint})

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
