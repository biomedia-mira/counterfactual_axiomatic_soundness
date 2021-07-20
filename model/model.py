from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.lax import stop_gradient

from components.classifier import calc_accuracy, calc_cross_entropy, classifier
from components.f_gan import f_gan
from components.typing import ApplyFn, Array, InitFn, PRNGKey, Shape, StaxLayer
from model.modes import get_layers
from model.train import Model, Params, UpdateFn
from jax import partial


# mechanism that acts on image based on categorical parent variable
def mechanism(parent_name: str, parent_dims: Dict[str, int], layers: Iterable[StaxLayer]) -> Tuple[InitFn, ApplyFn]:
    extra_dims = sum(parent_dims.values()) + parent_dims[parent_name]
    net_init_fn, net_apply_fn = stax.serial(*layers)

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        return net_init_fn(rng, (*input_shape[:-1], input_shape[-1] + extra_dims))

    def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array) -> Array:
        def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
            return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

        parents_ = broadcast(jnp.concatenate([*parents.values(), do_parent], axis=-1), (*inputs.shape[:-1], extra_dims))
        return net_apply_fn(params, jnp.concatenate((inputs, parents_), axis=-1))

    return init_fn, apply_fn


def build_model(parent_dims: Dict[str, int],
                marginals: Dict[str, Array],
                input_shape: Shape,
                mode: str,
                img_decode_fn: Callable[[np.ndarray], np.ndarray],
                cycle: bool = True) -> Model:
    classifier_layers, f_divergence_layers, mechanism_layers = get_layers(mode, input_shape)
    parent_names = list(parent_dims.keys())
    classifiers = {p_name: classifier(dim, layers=classifier_layers) for p_name, dim in parent_dims.items()}
    divergences = {p_name: f_gan(mode='gan', layers=f_divergence_layers, trick_g=True) for p_name in parent_dims.keys()}
    mechanisms = {p_name: mechanism(p_name, parent_dims, layers=mechanism_layers) for p_name in parent_dims.keys()}
    # this can be updated in the future for sequence of interventions
    interventions = tuple((parent_name,) for parent_name in parent_names)

    def init_fn(rng: Array, input_shape: Shape) -> Params:
        classifier_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in classifiers.items()}
        divergence_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in divergences.items()}
        mechanism_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in mechanisms.items()}
        return classifier_params, divergence_params, mechanism_params

    def sample_parent_from_marginals(rng: PRNGKey, parent_name: str, batch_size: int) -> Array:
        dim = parent_dims[parent_name]
        return jnp.eye(dim)[jax.random.choice(rng, dim, shape=(batch_size,), p=marginals[parent_name])]

    def classify(params: Params, parent_name: str, image: Array, target: Array) -> Tuple[Array, Any]:
        (_, _classify), _classifier_params = classifiers[parent_name], params[0][parent_name]
        prediction = _classify(_classifier_params, image)
        cross_entropy, accuracy = calc_cross_entropy(prediction, target), calc_accuracy(prediction, target)
        return cross_entropy, {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    def assert_dist(params: Params, target_dist: str, inputs: Any, image: Array, parents: Dict[str, Array]) -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}

        for parent_name, parent in parents.items():
            cross_entropy, output[parent_name] = classify(stop_gradient(params), parent_name, image, parent)
            loss = loss + cross_entropy

        (_, _calc_divergence), _divergence_params = divergences[target_dist], params[1][target_dist]
        image_target_dist, _ = inputs[target_dist]
        divergence, disc_loss, gen_loss = _calc_divergence(_divergence_params, image_target_dist, image)
        loss = loss + gen_loss - disc_loss
        return loss, {**output, 'divergence': divergence}

    def intervene(params: Params, image: Array, parents: Dict[str, Array], do_parent_name: str, rng: PRNGKey,
                  inputs: Any, source_dist: str, target_dist: str) -> Tuple[Array, Any]:

        (_, _transform), _transform_params = mechanisms[do_parent_name], params[2][do_parent_name]

        do_parent = sample_parent_from_marginals(rng, do_parent_name, batch_size=image.shape[0])
        do_image, do_parents = _transform(_transform_params, image, parents, do_parent)
        do_parents = {**parents, do_parent_name: do_parent}
        loss, output = assert_dist(params, do_image, do_parents, inputs, target_dist)

        order = jnp.argsort(jnp.argmax(do_parent, axis=-1))
        output = {**output, 'image': image[order], 'do_image': do_image[order]}

        if cycle:
            image_cycle = _transform(_transform_params, do_image, do_parents, parents[do_parent_name])
            loss_c, output_c = assert_dist(params, image_cycle, parents, inputs, source_dist)
            output = {'forward': output, 'cycle': {**output_c, 'undo_image': image_cycle[order]}}
            loss = loss + loss_c

        return loss, output

    def apply_fn(params: Params, inputs: Any, rng: Array) -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}

        # Train the classifiers on unconfounded data
        for parent_name in parent_names:
            image, parents = inputs[parent_name]
            cross_entropy, output[parent_name] = classify(params, parent_name, image, parents[parent_name])
            loss = loss + cross_entropy

        # Transform the confounded data into to the target (unconfounded) distributions
        for intervention in interventions:
            (image, parents), source_dist = inputs['joint'], 'joint'
            for i, do_parent_name in enumerate(intervention):
                target_dist = intervention[i] if i == 0 else set(intervention[:i + 1])
                l, o = intervene(params, image, parents, do_parent_name, rng, inputs, source_dist, target_dist)
                loss = loss + l
                output['do_' + '_'.join(intervention[:i + 1])] = o
                source_dist = target_dist

        return loss, output

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn]:
        # opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.0001, mass=0.5)
        opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 0.0002, b1=0.5)

        @jax.jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: Array) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = jax.value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
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

    return init_fn, apply_fn, init_optimizer_fn, accumulate_output, log_output
