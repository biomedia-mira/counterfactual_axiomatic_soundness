from typing import Any, Callable, Dict, FrozenSet, Iterable, Tuple

import jax.lax
import jax.numpy as jnp
from jax import jit, random, value_and_grad
from jax.experimental import optimizers
from jax.experimental.optimizers import OptimizerState, ParamsFn
from jax.lax import stop_gradient

from components.f_gan_refactor import f_gan
from components.stax_extension import Array, calc_accuracy, calc_cross_entropy, classifier, InitFn, PRNGKey, Shape, \
    StaxLayer
from model.train import Model, Params, UpdateFn

CriticFn = Callable[[Tuple[Array, Array]], Array]  # [[image, parents], score]
MechanismFn = Callable[
    [Array, Array, Array, Array], Tuple[Array, Array]]  # [[image, parent, do_parent, do_noise], [do_image, noise]]



def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


def l2(x: Array, y: Array) -> Array:
    return jnp.mean(jnp.sqrt(jnp.mean(jnp.power(x - y, 2), axis=tuple(range(1, x.ndim)))))


def l1(x: Array, y: Array) -> Array:
    return jnp.mean(jnp.sqrt(jnp.mean(jnp.abs(x - y), axis=tuple(range(1, x.ndim)))))


def condition_on_parents(parent_dims: Dict[str, int]) -> StaxLayer:
    def init_fn(rng: PRNGKey, shape: Shape) -> Tuple[Shape, Params]:
        return (*shape[:-1], shape[-1] + sum(parent_dims.values())), ()

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Array:
        image, parents = inputs
        shape = (*image.shape[:-1], sum(parent_dims.values()))
        _parents = jnp.concatenate([parents[key] for key in parent_dims.keys()], axis=-1)
        return jnp.concatenate((image, broadcast(_parents, shape)), axis=-1)

    return init_fn, apply_fn


def classifier_wrapper(num_classes: int, layers: Iterable[StaxLayer]) -> Model:
    init_fn, classify_fn = classifier(num_classes, layers)

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Tuple[Array, Dict[str, Array]]:
        image, target = inputs
        prediction = classify_fn(params, image)
        cross_entropy, accuracy = calc_cross_entropy(prediction, target), calc_accuracy(prediction, target)
        return jnp.mean(cross_entropy), {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn, Callable[[OptimizerState], Callable]]:
        opt_init, opt_update, get_params = optimizers.adam(step_size=5e-4, b1=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: PRNGKey) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng=rng)
            opt_state = opt_update(i, grads, opt_state)
            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    return init_fn, apply_fn, init_optimizer_fn


def build_model(parent_dims: Dict[str, int],
                marginal_dist: Array,
                do_parent_name: str,
                classifiers: Dict[str, Callable],
                critic_fn: Callable[..., Tuple[InitFn, CriticFn]],
                mechanism_fn: Callable[..., Tuple[InitFn, MechanismFn]],
                noise_dim: int,
                mode: int = 0) -> Tuple[Model, Any]:
    parent_names = list(parent_dims.keys())
    divergence_init_fn, divergence_apply_fn = f_gan(critic_fn, mode='gan', trick_g=True)
    mechanisms_init_fn, mechanism_apply_fn = mechanism_fn(do_parent_name, parent_dims, noise_dim)

    source_dist: FrozenSet[str] = frozenset(parent_names) if mode == 0 else frozenset()
    target_dist = frozenset(parent_names) if mode == 0 else frozenset((do_parent_name,))

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Params:
        _, divergence_params = divergence_init_fn(rng, input_shape)
        _, mechanism_params = mechanisms_init_fn(rng, input_shape)
        return (), (divergence_params, mechanism_params)

    def sample_parent_from_marginal(rng: PRNGKey, batch_size: int) -> Tuple[Array, Array]:
        parent_dim = marginal_dist.shape[0]
        _do_parent = random.choice(rng, parent_dim, shape=(batch_size,), p=marginal_dist)
        return jnp.eye(parent_dim)[_do_parent], jnp.argsort(_do_parent)

    def assert_dist(divergence_params: Params, inputs_target_dist: Any, image: Array, parents: Dict[str, Array]) \
            -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}
        # Assert the parents are correct
        for parent_name, parent in parents.items():
            cross_entropy, output[parent_name] = classifiers[parent_name]((image, parent))
            loss = loss + cross_entropy
        divergence, critic_loss, generator_loss = divergence_apply_fn(divergence_params, inputs_target_dist,
                                                                      (image, parents))
        loss = loss + critic_loss + generator_loss
        return loss, {**output, 'critic_loss': critic_loss[jnp.newaxis], 'generator_loss': generator_loss,
                      'divergence': divergence[jnp.newaxis]}

    def apply_fn(params: Params, inputs: Any, rng: PRNGKey) -> Tuple[Array, Any]:
        divergence_params, mechanism_params = params
        (image, parents) = inputs[source_dist]

        do_parent, order = sample_parent_from_marginal(rng, batch_size=image.shape[0])
        do_noise = random.uniform(rng, shape=(image.shape[0], noise_dim))

        do_image, noise = mechanism_apply_fn(mechanism_params, image, parents, do_parent, do_noise)
        do_parents = {**parents, do_parent_name: do_parent}
        loss, assertion_output = assert_dist(params, inputs, do_image, do_parents)

        # identity constraint
        image_same, noise_same = mechanism_apply_fn(mechanism_params, image, parents, parents[do_parent_name], noise)
        id_constraint = l2(image, image_same) + l2(do_noise, noise_same)

        # cycle constraint
        image_cycle, do_noise_cycle = mechanism_apply_fn(mechanism_params, stop_gradient(do_image), do_parents,
                                                         parents[do_parent_name], noise)
        cycle_constraint = l2(image, image_cycle) + l2(do_noise, do_noise_cycle)

        loss = loss + id_constraint + cycle_constraint

        output = {'loss': loss[jnp.newaxis],
                  'image': image[order],
                  'do_image': do_image[order],
                  'image_same': image_same[order],
                  'image_cycle': image_cycle[order],
                  'assertion_output': assertion_output,
                  'id_constraint': id_constraint,
                  'cycle_constraint': cycle_constraint,
                  }

        return loss, output

    def schedule(step: int, base_lr: float = 5e-4, gamma: float = .999) -> float:
        return base_lr * gamma ** step

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizers.adam(step_size=schedule, b1=0.5)
        key = f'do_{do_parent_name}'

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: PRNGKey) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            opt_state = opt_update(i, grads, opt_state)
            return opt_state, loss, outputs

        init_opt_state = opt_init(params)

        return init_opt_state, update, get_params

    return (init_fn, apply_fn, init_optimizer_fn), (divergence_apply_fn, mechanism_apply_fn)
