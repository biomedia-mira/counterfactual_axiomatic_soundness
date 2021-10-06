from typing import Any, Callable, Dict, FrozenSet, Iterable, Tuple

import jax.numpy as jnp
from jax import jit, random, value_and_grad
from jax.experimental import optimizers
from jax.experimental.optimizers import OptimizerState, ParamsFn
from jax.lax import stop_gradient
from more_itertools import powerset
from jax.experimental.optimizers import l2_norm
from components.classifier import calc_accuracy, calc_cross_entropy, classifier
from components.f_gan import f_gan
from components.typing import Array, InitFn, PRNGKey, Shape, StaxLayer
from model.train import Model, Params, UpdateFn

MechanismFn = Callable[[str, Dict[str, int], int],
                       Tuple[InitFn, Callable[[Params, Array, Dict[str, Array], Array, Array], Tuple[Array, Array]]]]


def l2(x: Array, y: Array) -> Array:
    return jnp.mean(jnp.sqrt(jnp.mean(jnp.power(x - y, 2), axis=tuple(range(1, x.ndim)))))


def build_model(parent_dims: Dict[str, int],
                marginals: Dict[str, Array],
                interventions: Iterable[Tuple[str, ...]],
                disc_layers: Iterable[StaxLayer],
                mechanism_fn: MechanismFn,
                noise_dim: int,
                mode: int = 0) -> Model:
    parent_names = list(parent_dims.keys())
    classifiers = {p_name: classifier(dim, disc_layers) for p_name, dim in parent_dims.items()}
    divergences = {frozenset(key): f_gan(disc_layers, mode='gan', trick_g=True) for key in powerset(parent_names)}
    mechanisms = {p_name: mechanism_fn(p_name, parent_dims, noise_dim) for p_name in parent_names}

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Params:
        classifier_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in classifiers.items()}
        divergence_params = {key: _init_fn(rng, input_shape)[1] for key, (_init_fn, _) in divergences.items()}
        mechanism_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in mechanisms.items()}
        return classifier_params, divergence_params, mechanism_params

    def sample_parent_from_marginals(rng: PRNGKey, parent_name: str, batch_size: int) -> Tuple[Array, Array]:
        _do_parent = random.choice(rng, parent_dims[parent_name], shape=(batch_size,), p=marginals[parent_name])
        return jnp.eye(parent_dims[parent_name])[_do_parent], jnp.argsort(_do_parent)

    def classify(params: Params, parent_name: str, image: Array, target: Array) -> Tuple[Array, Any]:
        (_, _classify), _classifier_params = classifiers[parent_name], params[0][parent_name]
        prediction = _classify(_classifier_params, image)
        cross_entropy, accuracy = calc_cross_entropy(prediction, target), calc_accuracy(prediction, target)
        return cross_entropy, {'cross_entropy': cross_entropy, 'accuracy': accuracy}

    def transform(params: Params, do_parent_name: str, image: Array, parents: Dict[str, Array], do_parent: Array,
                  do_noise: Array) -> Tuple[Array, Dict[str, Array], Array]:
        (_, _transform), _transform_params = mechanisms[do_parent_name], params[2][do_parent_name]
        do_image, noise = _transform(_transform_params, image, parents, do_parent, do_noise)
        do_parents = {**parents, do_parent_name: do_parent}
        return do_image, do_parents, noise

    def assert_dist(params: Params, target_dist: FrozenSet[str], inputs: Any, image: Array, parents: Dict[str, Array]) \
            -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}
        # Assert the parents are correct
        for parent_name, parent in parents.items():
            cross_entropy, output[parent_name] = classify(stop_gradient(params), parent_name, image, parent)
            loss = loss + cross_entropy
        # Assert the images come from the target distribution
        (_, _calc_divergence), _divergence_params = divergences[target_dist], params[1][target_dist]
        image_target_dist, parents_target_dist = inputs[target_dist]
        _divergence, disc_loss, gen_loss = _calc_divergence(_divergence_params, image_target_dist, image)
        loss = loss + gen_loss - disc_loss
        return loss, {**output, 'divergence': _divergence}

    def intervene(params: Params, source_dist: FrozenSet[str], target_dist: FrozenSet[str], do_parent_name: str,
                  inputs: Any, image: Array, parents: Dict[str, Array], rng: PRNGKey) \
            -> Tuple[Array, Array, Array, Any]:

        do_parent, order = sample_parent_from_marginals(rng, do_parent_name, batch_size=image.shape[0])
        do_noise = random.uniform(rng, shape=(image.shape[0], noise_dim)) * .1
        do_image, do_parents, noise = transform(params, do_parent_name, image, parents, do_parent, do_noise)
        loss, _output = assert_dist(params, target_dist, inputs, do_image, do_parents)
        output = {'image': image[order], 'do_image': do_image[order], 'forward': _output}

        # cycle
        image_cycle, _, do_noise_cycle = transform(params, do_parent_name, stop_gradient(do_image), do_parents,
                                                   parents[do_parent_name], noise)
        loss_cycle, _output = assert_dist(params, source_dist, inputs, image_cycle, parents)

        l2_image = l2(image, image_cycle)
        l2_do_noise = l2(do_noise, do_noise_cycle)

        loss = loss + loss_cycle + l2_image + l2_do_noise
        output.update({'image_cycle': image_cycle[order], 'cycle': _output, 'l2_image': l2_image,
                       'l2_do_noise': l2_do_noise})

        return do_image, do_parents, loss, output

    def apply_fn(params: Params, inputs: Any, rng: PRNGKey) -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}

        # Step classifiers
        for parent_name in parent_names:
            image, parents = inputs[frozenset(parent_names)] if mode == 0 else inputs[frozenset((parent_name,))]

            cross_entropy, output[parent_name] = classify(params, parent_name, image, parents[parent_name])
            loss = loss + cross_entropy

        # Perform interventions
        for intervention in interventions:
            source_dist: FrozenSet[str] = frozenset(parent_names) if mode == 0 else frozenset()
            (image, parents) = inputs[source_dist]
            for i, do_parent_name in enumerate(intervention):
                target_dist = frozenset(parent_names) if mode == 0 else frozenset(intervention[:i + 1])
                key = 'do_' + '_'.join(intervention[:i + 1])

                do_image, do_parents, _loss, output[key] = intervene(params, source_dist, target_dist, do_parent_name,
                                                                     inputs, image, parents, rng)
                loss = loss + _loss
                image, parents, source_dist = do_image, do_parents, target_dist
        # loss = loss + l2_norm(params)
        output['total_loss'] = loss
        return loss, output

    def schedule(step: int, base_lr: float = 5e-4, gamma: float = .997) -> float:
        return base_lr * gamma ** step

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizers.adam(step_size=schedule, b1=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: PRNGKey) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            opt_state = opt_update(i, grads, opt_state)

            return opt_state, loss, outputs

        init_opt_state = opt_init(params)

        return init_opt_state, update, get_params

    return init_fn, apply_fn, init_optimizer_fn
