from typing import Dict, Tuple, cast

import jax
import jax.numpy as jnp
import jax.ops
import numpy as np
from jax import jit
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.experimental.stax import Dense, Flatten, LogSoftmax
from jax.lax import stop_gradient

from components.f_divergence import f_divergence
from model.modes import get_layers
from trainer.training import ApplyFn, InitFn, InitOptimizerFn, Params, Tree, UpdateFn


def classifier(num_classes: int, layers):
    return stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax)


def calc_accuracy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return cast(jnp.ndarray, jnp.equal(jnp.argmax(pred, axis=-1), jnp.argmax(target, axis=-1)))


def calc_cross_entropy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return cast(jnp.ndarray, -jnp.mean(jnp.sum(pred * target, axis=-1)))


def classifier_loss(classifier_apply_fun, targets: jnp.ndarray):
    def wrapper(params: Params, inputs: jnp.ndarray, **kwargs):
        prediction = classifier_apply_fun(params, inputs)
        cross_entropy, accuracy = calc_cross_entropy(prediction, targets), calc_accuracy(prediction, targets)
        return cross_entropy, {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    return wrapper


# mechanism that acts on image based on categorical parent variable
def mechanism(parent_name: str, parent_dims: Dict[str, int], layers):
    extra_dims = sum(parent_dims.values()) + parent_dims[parent_name]
    net_init_fun, net_apply_fun = stax.serial(*layers)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        return net_init_fun(rng, (*input_shape[:-1], input_shape[-1] + extra_dims))

    def apply_fun(params: Params, inputs: jnp.ndarray, parents: Dict[str, jnp.ndarray], do_parent: jnp.ndarray):
        def broadcast(array: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
            return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

        parents_ = broadcast(jnp.concatenate([*parents.values(), do_parent], axis=-1), (*inputs.shape[:-1], extra_dims))
        return net_apply_fun(params, jnp.concatenate((inputs, parents_), axis=-1))

    return init_fun, apply_fun


def build_model(parent_dims: Dict[str, int],
                marginals: Dict[str, np.ndarray],
                input_shape: Tuple[int, ...],
                mode: str) -> Tuple[InitFn, ApplyFn, InitOptimizerFn]:
    classifier_layers, f_divergence_layers, mechanism_layers = get_layers(mode, input_shape)
    parent_names = list(parent_dims.keys())
    classifiers = {p_name: classifier(dim, layers=classifier_layers) for p_name, dim in parent_dims.items()}
    divergences = {p_name: f_divergence(mode='kl', layers=f_divergence_layers) for p_name in parent_dims.keys()}
    mechanisms = {p_name: mechanism(p_name, parent_dims, layers=mechanism_layers) for p_name in parent_dims.keys()}
    # this can be updated in the future for sequence of interventions
    interventions = tuple((parent_name,) for parent_name in parent_names)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]) -> Params:
        classifier_params = {p_name: _init_fun(rng, input_shape) for p_name, (_init_fun, _) in classifiers.items()}
        divergence_params = {p_name: _init_fun(rng, input_shape) for p_name, (_init_fun, _) in divergences.items()}
        mechanism_params = {p_name: _init_fun(rng, input_shape) for p_name, (_init_fun, _) in mechanisms.items()}
        return classifier_params, divergence_params, mechanism_params

    def classifier_step(classifier_params: Params, inputs: Tree[jnp.ndarray]) -> Tuple[jnp.ndarray, Tree[jnp.ndarray]]:
        loss, output = jnp.zeros(()), {}
        for parent_name in parent_names:
            image, parents = inputs[parent_name]
            (_, _apply_fun), _params = classifiers[parent_name], classifier_params[parent_name]
            cross_entropy, output[parent_name] = classifier_loss(_apply_fun, parents[parent_name])(_params, image)
            loss = loss + cross_entropy
        return loss, output

    def transform(mechanism_params: Params, image: jnp.ndarray, parents: Dict[str, jnp.ndarray], parent_name: str,
                  rng: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Tree[jnp.ndarray]]:
        (_, _apply_fun), _params = mechanisms[parent_name], mechanism_params[parent_name]
        _do_parent = jax.random.choice(rng, parent_dims[parent_name], shape=image.shape[:1], p=marginals[parent_name])
        do_parent = jnp.eye(parent_dims[parent_name])[_do_parent]
        order = jnp.argsort(_do_parent)
        do_image = _apply_fun(_params, image, parents, do_parent)
        return do_image, {**parents, parent_name: do_parent}, {'image': image[order], 'do_image': do_image[order]}

    def apply_fun(params: Params, inputs: Tree[np.ndarray], rng: jnp.ndarray) -> Tuple[jnp.ndarray, Tree[jnp.ndarray]]:
        loss, output = jnp.zeros(()), {}
        classifier_params, divergence_params, mechanism_params = params

        image, parents = inputs['joint']
        for intervention in interventions:
            do_image, do_parents = image, parents
            for i, parent_name in enumerate(intervention):
                # Transform image
                do_image, do_parents, output_t = transform(mechanism_params, do_image, do_parents, parent_name, rng)
                # Ensure transformed inputs has the correct parents
                cross_entropy, output_c = classifier_step(stop_gradient(classifier_params), (do_image, do_parents))
                # Ensure image comes from correct distribution
                target_dist = intervention[i] if i == 0 else set(intervention[:i + 1])
                image_target_dist, _ = inputs[target_dist]
                (_, _apply_fun), _params = divergences[target_dist], divergence_params[target_dist]
                divergence_gen = _apply_fun(stop_gradient(_params), image_target_dist, do_image)
                divergence_disc = _apply_fun(_params, image_target_dist, stop_gradient(do_image))
                # Mechanism minimises cross-entropy and divergence, discriminator maximises divergence
                loss += cross_entropy + divergence_gen - divergence_disc
                output['do_' + '_'.join(intervention[:i + 1])] = {**output_t, **output_c, 'divergence': divergence_gen}

        return loss, output

    def init_optimizer_fun(params: Params) -> Tuple[OptimizerState, UpdateFn]:
        opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.0001, mass=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Tree[np.ndarray], rng: jnp.ndarray) \
                -> Tuple[OptimizerState, jnp.ndarray, Tree[jnp.ndarray]]:
            classifier_params, divergence_params, mechanism_params = get_params(opt_state)
            (loss_c, outputs_c), grads_c = jax.value_and_grad(classifier_step, has_aux=True)(classifier_params, inputs)
            (loss_i, outputs_i), grads_i = jax.value_and_grad(apply_fun, has_aux=True)(params, inputs, rng)
            opt_state = opt_update(i, (grads_c, *grads_i[1:]), opt_state)
            return opt_state, loss_c + loss_i, {**outputs_i, 'classifiers': outputs_c}

        opt_state = opt_init(params)

        return opt_state, update

    return init_fun, apply_fun, init_optimizer_fun
