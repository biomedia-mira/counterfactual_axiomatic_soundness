from typing import Any, Callable, Dict, FrozenSet, Iterable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random, value_and_grad
from jax.experimental import optimizers
from jax.experimental.optimizers import OptimizerState, ParamsFn
from jax.lax import stop_gradient

from components.f_gan import f_gan
from components.stax_extension import Array, calc_accuracy, calc_cross_entropy, classifier, InitFn, PRNGKey, Shape, \
    StaxLayer
from model.train import Model, Params, UpdateFn


def l2(x: Array, y: Array) -> Array:
    return jnp.mean(jnp.sqrt(jnp.mean(jnp.power(x - y, 2), axis=tuple(range(1, x.ndim)))))


# [[[image, parents]], score]
ClassifierFn = Callable[[Tuple[Array, Array]], Array]
# [[params, [image, parents]], score]
CriticFn = Callable[[Params, Tuple[Array, Array]], Array]
# [[params, image, parents, do_parent, do_noise], [do_image, noise]]
MechanismFn = Callable[[Params, Array, Dict[str, Array], Array, Array], Tuple[Array, Array]]


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
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs)
            opt_state = opt_update(i, grads, opt_state)
            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    return init_fn, apply_fn, init_optimizer_fn


def model_wrapper(source_dist: FrozenSet[str],
                  do_parent_name: str,
                  marginal_dist: Array,
                  classifiers: Dict[str, ClassifierFn],
                  critic: Tuple[InitFn, CriticFn],
                  mechanism: Tuple[InitFn, MechanismFn],
                  noise_dim: int) -> Tuple[Model, Any]:
    target_dist = source_dist.union((do_parent_name,))
    divergence_init_fn, divergence_apply_fn = f_gan(critic, mode='gan', trick_g=True)
    mechanisms_init_fn, mechanism_apply_fn = mechanism

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Params:
        _, divergence_params = divergence_init_fn(rng, input_shape)
        _, mechanism_params = mechanisms_init_fn(rng, input_shape)
        return (), (divergence_params, mechanism_params)

    def sample_parent_from_marginal(rng: PRNGKey, batch_size: int) -> Tuple[Array, Array]:
        parent_dim = marginal_dist.shape[0]
        _do_parent = random.choice(rng, parent_dim, shape=(batch_size,), p=marginal_dist)
        return jnp.eye(parent_dim)[_do_parent], jnp.argsort(_do_parent)

    def assert_dist(divergence_params: Params, inputs: Any, image: Array, parents: Dict[str, Array]) \
            -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}
        # Assert the parents are correct
        for parent_name, parent in parents.items():
            cross_entropy, output[parent_name] = classifiers[parent_name]((image, parent))
            loss = loss + cross_entropy
        div_loss, div_output = divergence_apply_fn(divergence_params, inputs[target_dist], (image, parents))
        loss = loss + div_loss
        return loss, {**output, **div_output}

    def apply_fn(divergence_params: Params, inputs: Any, rng: PRNGKey) -> Tuple[Array, Any]:
        divergence_params, mechanism_params = divergence_params
        (image, parents) = inputs[source_dist]
        k1, k2 = jax.random.split(rng)
        do_parent, order = sample_parent_from_marginal(k1, batch_size=image.shape[0])
        do_noise = random.uniform(k2, shape=(image.shape[0], noise_dim))

        do_image, noise = mechanism_apply_fn(mechanism_params, image, parents, do_parent, do_noise)
        do_parents = {**parents, do_parent_name: do_parent}
        loss, assertion_output = assert_dist(divergence_params, inputs, do_image, do_parents)

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
                  'id_constraint': id_constraint,
                  'cycle_constraint': cycle_constraint,
                  **assertion_output
                  }

        return loss, output

    def schedule(step: int, base_lr: float = 5e-4, gamma: float = .999) -> float:
        return base_lr * gamma ** step

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizers.adam(step_size=schedule, b1=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: PRNGKey) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            opt_state = opt_update(i, grads, opt_state)
            return opt_state, loss, outputs

        return opt_init(params), update, get_params

    return (init_fn, apply_fn, init_optimizer_fn), (divergence_apply_fn, mechanism_apply_fn)
