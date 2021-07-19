from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.lax import stop_gradient

from components.classifier import calc_accuracy, calc_cross_entropy, classifier
from components.f_gan import f_gan
from components.stax_layers import ApplyFn, Array, InitFn, Shape, StaxLayer
from model.modes import get_layers
from model.train import Model, Params, UpdateFn


# mechanism that acts on image based on categorical parent variable
def mechanism(parent_name: str, parent_dims: Dict[str, int], layers: Iterable[StaxLayer]) -> Tuple[InitFn, ApplyFn]:
    extra_dims = sum(parent_dims.values()) + parent_dims[parent_name]
    net_init_fun, net_apply_fun = stax.serial(*layers)

    def init_fun(rng: Array, input_shape: Tuple[int, ...]) -> Tuple[Shape, Params]:
        return net_init_fun(rng, (*input_shape[:-1], input_shape[-1] + extra_dims))

    def apply_fun(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array) -> Array:
        def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
            return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

        parents_ = broadcast(jnp.concatenate([*parents.values(), do_parent], axis=-1), (*inputs.shape[:-1], extra_dims))
        return net_apply_fun(params, jnp.concatenate((inputs, parents_), axis=-1))

    return init_fun, apply_fun


def build_model(parent_dims: Dict[str, int],
                marginals: Dict[str, Array],
                input_shape: Shape,
                mode: str,
                img_decode_fn: Callable[[np.array], np.array]) -> Model:
    classifier_layers, f_divergence_layers, mechanism_layers = get_layers(mode, input_shape)
    parent_names = list(parent_dims.keys())
    classifiers = {p_name: classifier(dim, layers=classifier_layers) for p_name, dim in parent_dims.items()}
    divergences = {p_name: f_gan(mode='gan', layers=f_divergence_layers, trick_g=True) for p_name in parent_dims.keys()}
    mechanisms = {p_name: mechanism(p_name, parent_dims, layers=mechanism_layers) for p_name in parent_dims.keys()}
    # this can be updated in the future for sequence of interventions
    interventions = tuple((parent_name,) for parent_name in parent_names)

    def init_fun(rng: Array, input_shape: Shape) -> Params:
        classifier_params = {p_name: _init_fun(rng, input_shape)[1] for p_name, (_init_fun, _) in classifiers.items()}
        divergence_params = {p_name: _init_fun(rng, input_shape)[1] for p_name, (_init_fun, _) in divergences.items()}
        mechanism_params = {p_name: _init_fun(rng, input_shape)[1] for p_name, (_init_fun, _) in mechanisms.items()}
        return classifier_params, divergence_params, mechanism_params

    def classify(_apply_fun: ApplyFn, target: Array) -> Callable[[Params, Array], Tuple[Array, Dict[str, Array]]]:
        def wrapper(_params: Params, image: Array) -> Tuple[Array, Dict[str, Array]]:
            prediction = _apply_fun(_params, image)
            cross_entropy, accuracy = calc_cross_entropy(prediction, target), calc_accuracy(prediction, target)
            return cross_entropy, {'cross_entropy': cross_entropy, 'accuracy': accuracy}

        return wrapper

    def transform(mechanism_params: Params, image: Array, parents: Dict[str, Array], parent_name: str,
                  rng: Array) -> Tuple[Array, Dict[str, Array], Any]:
        (_, _apply_fun), _params = mechanisms[parent_name], mechanism_params[parent_name]
        _do_parent = jax.random.choice(rng, parent_dims[parent_name], shape=image.shape[:1], p=marginals[parent_name])
        do_parent = jnp.eye(parent_dims[parent_name])[_do_parent]
        order = jnp.argsort(_do_parent)
        do_image = _apply_fun(_params, image, parents, do_parent)
        return do_image, {**parents, parent_name: do_parent}, {'image': image[order], 'do_image': do_image[order]}

    def assert_dist(params: Params, do_image: Array, do_parents: Dict[str, Array], inputs: Any, target_dist: str) -> \
            Tuple[Array, Any]:
        classifier_params, divergence_params, mechanism_params = params
        loss, output = jnp.zeros(()), {}
        # Ensure image has the correct parents
        for parent_name in parent_names:
            (_, _apply_fun), _params = classifiers[parent_name], classifier_params[parent_name]
            cross_entropy, output[parent_name] = classify(_apply_fun, do_parents[parent_name])(stop_gradient(_params),
                                                                                               do_image)
            loss = loss + cross_entropy

        # Ensure image comes from correct distribution
        image_target_dist, _ = inputs[target_dist]
        (_, _apply_fun), _params = divergences[target_dist], divergence_params[target_dist]
        divergence, disc_loss, gen_loss = _apply_fun(_params, image_target_dist, do_image)

        # Mechanism minimises cross-entropy and divergence, discriminator maximises divergence
        loss = loss + gen_loss - disc_loss
        return loss, {**output, 'divergence': divergence}

    def apply_fun(params: Params, inputs: Any, rng: Array) -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}
        classifier_params, divergence_params, mechanism_params = params

        # Train the classifiers on unconfounded data
        for parent_name in parent_names:
            image, parents = inputs[parent_name]
            (_, _apply_fun), _params = classifiers[parent_name], classifier_params[parent_name]
            cross_entropy, output[parent_name] = classify(_apply_fun, parents[parent_name])(_params, image)
            loss = loss + cross_entropy

        # Transform the confounded data into to the target (unconfounded) distributions
        image, parents = inputs['joint']
        for intervention in interventions:
            do_image, do_parents = image, parents
            for i, do_parent_name in enumerate(intervention):
                target_dist = intervention[i] if i == 0 else set(intervention[:i + 1])
                do_image, do_parents, output_t = transform(mechanism_params, do_image, do_parents, do_parent_name, rng)
                loss_a, output_a = assert_dist(params, do_image, do_parents, inputs, target_dist)
                loss = loss + loss_a
                output['do_' + '_'.join(intervention[:i + 1])] = {**output_t, **output_a}

        return loss, output

    def init_optimizer_fun(params: Params) -> Tuple[OptimizerState, UpdateFn]:
        # opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.0001, mass=0.5)
        opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 0.0002, b1=0.5)

        @jax.jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: Array) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = jax.value_and_grad(apply_fun, has_aux=True)(get_params(opt_state), inputs, rng)
            opt_state = opt_update(i, grads, opt_state)

            return opt_state, loss, outputs

        opt_state = opt_init(params)

        return opt_state, update

    # evaluation
    def update_value(value: Array, new_value: Array) -> Array:
        if value.size == 1:
            return value + new_value
        elif value.ndim == 1:
            return jnp.concatenate((value, new_value))
        else:
            return new_value

    def close_value(value: Array) -> Array:
        if value.size == 1:
            return value
        elif value.ndim == 1:
            return jnp.mean(value)
        else:
            return img_decode_fn(value)

    def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
        to_cpu = jax.partial(jax.device_put, device=jax.devices('cpu')[0])
        new_output = jax.tree_map(to_cpu, new_output)
        return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)

    def log_output(output: Any) -> Any:
        return jax.tree_map(close_value, output)

    return init_fun, apply_fun, init_optimizer_fun, accumulate_output, log_output
