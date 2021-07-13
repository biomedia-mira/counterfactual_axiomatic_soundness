from typing import Callable, Dict, List, Tuple, Union, cast

import jax
import jax.numpy as jnp
import jax.ops
import numpy as np
from jax import jit
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.experimental.stax import Dense, Flatten, LogSoftmax
from jax.lax import stop_gradient

from trainer.training import ApplyFn, InitFn, InitOptimizerFn, Params, Tree, UpdateFn
from components.f_divergence import f_divergence
from model.modes import get_layers


def classifier(num_classes: int, layers):
    return stax.serial(*layers, Flatten, Dense(num_classes), LogSoftmax)


# mechanism that acts on image based on categorical parent variable
def mechanism(parent_name: str,
              parent_dims: Dict[str, int],
              layers):
    extra_dims = sum(parent_dims.values()) + parent_dims[parent_name]
    net_init_fun, net_apply_fun = stax.serial(*layers)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        return net_init_fun(rng, (*input_shape[:-1], input_shape[-1] + extra_dims))

    def apply_fun(params: List[Tuple[jnp.ndarray, ...]],
                  inputs: jnp.ndarray,
                  parents: Dict[str, jnp.ndarray],
                  do_parent: jnp.ndarray):
        def broadcast(array: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
            return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)

        parents_ = broadcast(jnp.concatenate([*parents.values(), do_parent], axis=-1), (*inputs.shape[:-1], extra_dims))
        return net_apply_fun(params, jnp.concatenate((inputs, parents_), axis=-1))

    return init_fun, apply_fun


def calc_accuracy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return cast(jnp.ndarray, jnp.equal(jnp.argmax(pred, axis=-1), jnp.argmax(target, axis=-1)))


def calc_cross_entropy(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return cast(jnp.ndarray, -jnp.mean(jnp.sum(pred * target, axis=-1)))


def build_model(parent_dims: Dict[str, int],
                marginals: Dict[str, np.ndarray],
                input_shape: Tuple[int, ...],
                mode: str) -> Tuple[InitFn, ApplyFn, InitOptimizerFn]:
    classifier_layers, f_divergence_layers, mechanism_layers = get_layers(mode, input_shape)
    parent_names = list(parent_dims.keys())
    components: Dict[str, Dict[str, Tuple[Callable, Callable]]] = {key: {} for key in parent_names}
    for parent_name_, dim in parent_dims.items():
        components[parent_name_]['classifier'] = classifier(dim, layers=classifier_layers)
        components[parent_name_]['divergence'] = f_divergence(mode='kl', layers=f_divergence_layers)
        components[parent_name_]['mechanism'] = mechanism(parent_name_, parent_dims, layers=mechanism_layers)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]) -> Params:
        params: Params = {key: {} for key in parent_names}
        for parent_name, funcs in components.items():
            for component, (_init_fun, _) in funcs.items():
                _, params[parent_name][component] = _init_fun(rng, input_shape)

        return params

    def classifiers_step(params: Params, inputs: Tree[jnp.ndarray]) -> Tuple[jnp.ndarray, Tree[jnp.ndarray]]:
        loss, output = jnp.zeros(()), {}
        for parent_name in parent_names:
            image_uc, parents_uc = inputs[parent_name]
            _, classifier_apply_fun = components[parent_name]['classifier']
            prediction_uc = classifier_apply_fun(params[parent_name]['classifier'], image_uc)
            cross_entropy_uc = calc_cross_entropy(prediction_uc, parents_uc[parent_name])
            accuracy_uc = calc_accuracy(prediction_uc, parents_uc[parent_name])
            output.update({parent_name: {'cross_entropy': cross_entropy_uc, 'accuracy': accuracy_uc}})
            loss = loss + cross_entropy_uc
        return loss, output

    def transform(params: Params, image: jnp.ndarray, parents: Dict[str, jnp.ndarray], parent_name: str,
                  rng: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Tree[jnp.ndarray]]:
        _, mechanism_apply_fun = components[parent_name]['mechanism']
        do_parent = jax.random.choice(rng, parent_dims[parent_name], shape=image.shape[:1], p=marginals[parent_name])
        order = jnp.argsort(do_parent)
        do_parent_one_hot = jnp.eye(parent_dims[parent_name])[do_parent]
        do_parents = parents.copy()
        do_parents[parent_name] = do_parent_one_hot
        do_image = mechanism_apply_fun(params[parent_name]['mechanism'], image, parents, do_parent_one_hot)
        return do_image, do_parents, {'image': image[order], 'do_image': do_image[order]}

    def discriminator_step(params: Params,
                           inputs: Tree[jnp.ndarray],
                           do_image: jnp.ndarray,
                           do_parents: Dict[str, jnp.ndarray],
                           target_dist_key: Union[str, Tuple[str, ...]]) -> Tuple[jnp.ndarray, Tree[jnp.ndarray]]:
        loss, output = jnp.zeros(()), {}
        for parent_name in parent_names:
            _, classifier_apply_fun = components[parent_name]['classifier']
            prediction = jit(classifier_apply_fun)(stop_gradient(params[parent_name]['classifier']), do_image)
            cross_entropy = calc_cross_entropy(prediction, do_parents[parent_name])
            accuracy = calc_accuracy(prediction, do_parents[parent_name])
            output.update({parent_name: {'cross_entropy': cross_entropy, 'accuracy': accuracy}})
            loss = loss + cross_entropy

        assert isinstance(target_dist_key, str)
        image_target_dist, _ = inputs[target_dist_key]
        _, divergence_apply_fun = components[target_dist_key]['divergence']
        divergence = divergence_apply_fun(params[target_dist_key]['divergence'], image_target_dist, do_image)
        loss = loss + divergence
        output.update({'divergence': divergence})

        return loss, output

    def apply_fun(params: Params, inputs: Tree[np.ndarray]) -> Tuple[jnp.ndarray, Tree[jnp.ndarray]]:
        loss, output = jnp.zeros(()), {}

        # Train classifiers with unconfounded distributions
        loss_classifiers, output['classifiers'] = classifiers_step(params, inputs)
        loss = loss + loss_classifiers

        # this can be updated in the future for sequence of interventions
        interventions = tuple((parent_name,) for parent_name in parent_names)
        rng = jax.random.PRNGKey(4)

        image, parents = inputs['joint']
        for intervention in interventions:
            do_image, do_parents = image, parents
            for i, parent_name in enumerate(intervention):
                key = intervention[i] if i == 0 else tuple(intervention[:i + 1])
                # Transform the image
                do_image, do_parents, output_t = transform(params, do_image, do_parents, parent_name, rng)
                _, rng = jax.random.split(rng)
                # Ensure transformed inputs has the correct parents and is from correct distribution
                loss_discriminator, output_d = discriminator_step(params, inputs, do_image, do_parents, key)
                output[f'do_{key}'] = {**output_t, **output_d}
                loss = loss - loss_discriminator

        return loss, output

    def init_optimizer_fun(params: Params) -> Tuple[OptimizerState, UpdateFn]:
        opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.0001, mass=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Tree[np.ndarray]) \
                -> Tuple[OptimizerState, jnp.ndarray, Tree[jnp.ndarray]]:
            params = get_params(opt_state)
            (loss, outputs), grads = jax.value_and_grad(apply_fun, has_aux=True)(params, inputs)
            for parent in grads.keys():
                grads[parent]['mechanism'] = jax.tree_map(lambda x: x * -1, grads[parent]['mechanism'])
                # grads[parent]['divergence'] = jax.tree_map(lambda x: jnp.clip(a=x, a_min=-1, a_max=1), grads[parent]['divergence'])
                # grads[parent]['divergence'] = jax.tree_map(lambda x: x * .1, grads[parent]['divergence'])

            opt_state = opt_update(i, grads, opt_state)

            return opt_state, loss, outputs

        opt_state = opt_init(params)

        return opt_state, update

    return init_fun, apply_fun, init_optimizer_fun
