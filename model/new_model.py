from typing import Any, Callable, Dict, Iterable, Optional, Tuple, FrozenSet

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.experimental.stax import Flatten, Dense, Sigmoid
from jax.lax import stop_gradient
from more_itertools import powerset

from components.f_gan import f_gan
from components.typing import Array, PRNGKey, Shape, StaxLayer
from model.modes import get_layers
from model.train import Model, Params, UpdateFn


def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


def concat_inputs(inputs: Array, *inputs_to_broadcast: Array) -> Array:
    shape = (*inputs.shape[:-1], sum([_input.shape[-1] for _input in inputs_to_broadcast]))
    return jnp.concatenate((inputs, broadcast(jnp.concatenate(inputs_to_broadcast, axis=-1), shape=shape)), axis=-1)


def abduction(parent_dims: Dict[str, int], noise_dim: int, layers: Iterable[StaxLayer]) -> StaxLayer:
    _init_fn, _apply_fn = stax.serial(*layers, Flatten, Dense(noise_dim), Sigmoid)

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        extra_dim = sum(parent_dims.values())
        return _init_fn(rng, (*input_shape[:-1], input_shape[-1] + extra_dim))

    def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array]) -> Array:
        return _apply_fn(params, concat_inputs(inputs, *parents.values()))

    return init_fn, apply_fn


def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int, layers: Iterable[StaxLayer]) -> StaxLayer:
    _init_fn, _apply_fn = stax.serial(*layers)

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        extra_dim = sum(parent_dims.values()) + parent_dims[parent_name] + noise_dim
        return _init_fn(rng, (*input_shape[:-1], input_shape[-1] + extra_dim))

    def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array, noise: Array) -> Array:
        return _apply_fn(params, concat_inputs(inputs, *[*parents.values(), do_parent, noise]))

    return init_fn, apply_fn


def divergence(parent_dims: Dict[str, int], noise_dim: int, layers: Iterable[StaxLayer]) -> StaxLayer:
    _init_fn, _apply_fn = f_gan(layers, mode='gan', trick_g=True)

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        extra_dim = sum(parent_dims.values())
        return _init_fn(rng, (*input_shape[:-1], input_shape[-1] + extra_dim))

    def apply_fn(params: Params, inputs_r: Array, parents_r: Dict[str, Array], inputs_g: Array,
                 parents_g: Dict[str, Array]) -> Array:
        _divergence, disc_loss, gen_loss = _apply_fn(params, concat_inputs(inputs_r, *parents_r.values()),
                                                     concat_inputs(inputs_g, *parents_g.values()))
        loss = gen_loss - disc_loss
        return loss, _divergence, gen_loss

    return init_fn, apply_fn


def l2(x: Array, y: Array) -> Array:
    return jnp.mean(jnp.sqrt(jnp.mean(jnp.power(x - y, 2), axis=tuple(range(1, x.ndim)))))


def build_model(parent_dims: Dict[str, int], marginals: Dict[str, Array], input_shape: Shape, noise_dim: int,
                mode: str, img_decode_fn: Callable[[np.ndarray], np.ndarray], cycle: bool = False) -> Model:
    disc_layers, gen_layers = get_layers(mode, input_shape)
    parent_names = list(parent_dims.keys())
    abductions = {p_name: abduction(parent_dims, noise_dim, disc_layers) for p_name in parent_names}
    mechanisms = {p_name: mechanism(p_name, parent_dims, noise_dim, gen_layers) for p_name in parent_names}
    divergences = {frozenset(key): divergence(parent_dims, noise_dim, disc_layers) for key in powerset(parent_names)}
    # this can be updated in the future for sequence of interventions
    interventions = tuple((parent_name,) for parent_name in parent_names)  # uniqueness must be asserted

    def init_fn(rng: Array, input_shape: Shape) -> Params:
        abductor_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in abductions.items()}
        mechanism_params = {p_name: _init_fn(rng, input_shape)[1] for p_name, (_init_fn, _) in mechanisms.items()}
        divergence_params = {key: _init_fn(rng, input_shape)[1] for key, (_init_fn, _) in divergences.items()}
        return abductor_params, mechanism_params, divergence_params

    def sample_parent_from_marginals(rng: PRNGKey, parent_name: str, batch_size: int) -> Tuple[Array, Array]:
        _do_parent = jax.random.choice(rng, parent_dims[parent_name], shape=(batch_size,), p=marginals[parent_name])
        return jnp.eye(parent_dims[parent_name])[_do_parent], jnp.argsort(_do_parent)

    def intervene(params: Params, source_dist: FrozenSet[str], target_dist: FrozenSet[str], do_parent_name: str,
                  inputs: Any, image: Array, parents: Dict[str, Array], rng: PRNGKey) \
            -> Tuple[Array, Array, Array, Any]:

        (_, abduct), _abduct_params = abductions[do_parent_name], params[0][do_parent_name]
        (_, transform), _transform_params = mechanisms[do_parent_name], params[1][do_parent_name]
        (_, calc_divergence), _divergence_params = divergences[target_dist], params[2][target_dist]

        do_parent, order = sample_parent_from_marginals(rng, do_parent_name, batch_size=image.shape[0])
        do_noise = jax.random.uniform(rng, shape=(image.shape[0], noise_dim))
        do_image = transform(_transform_params, image, parents, do_parent, do_noise)
        do_parents = {**parents, do_parent_name: do_parent}

        image_target_dist, parents_target_dist = inputs[target_dist]
        loss, _divergence,gen_loss = calc_divergence(_divergence_params, image_target_dist, parents_target_dist, do_image,
                                            do_parents)
        output = {'image': image[order], 'do_image': do_image[order], 'divergence': _divergence, 'loss': loss, 'gen_loss':gen_loss}

        # if cycle:
        #     noise = abduct(_abduct_params, image, parents)
        #     image_cycle, _ = transform(_transform_params, do_image, do_parents, do_parent, noise)
        #     image_target_dist, parents_target_dist = inputs[source_dist]
        #     loss_cycle, _divergence = calc_divergence(divergence_params, image_target_dist, parents_target_dist,
        #                                               image_cycle, parents)
        #     # noise_cycle = abduct(params, stop_gradient(image_cycle), stop_gradient(parents))
        #     do_noise_cycle = abduct(_abduct_params, stop_gradient(do_image), do_parents)
        #     l2_image = l2(image, image_cycle)
        #     # l2_noise = l2(noise, noise_cycle)
        #     l2_do_noise = l2(do_noise, do_noise_cycle)
        #
        #     loss += loss_cycle + l2_do_noise
        #     output.update(
        #         {'image_cycle': image_cycle[order], 'divergence_cycle': _divergence, 'l2_image': l2_image,
        #          'l2_do_noise': l2_do_noise})

        return do_image, do_parents, loss, output

    def apply_fn(params: Params, inputs: Any, rng: Array) -> Tuple[Array, Any]:
        loss, output = jnp.zeros(()), {}

        for intervention in interventions:
            source_dist: FrozenSet[str] = frozenset()  # empty set represents joint distribution
            (image, parents) = inputs[source_dist]
            for i, do_parent_name in enumerate(intervention):
                target_dist = frozenset(intervention[:i + 1])
                key = 'do_' + '_'.join(intervention[:i + 1])

                do_image, do_parents, _loss, output[key] = intervene(params, source_dist, target_dist, do_parent_name,
                                                                     inputs, image, parents, rng)
                loss += _loss

                image, parents, source_dist = do_image, do_parents, target_dist

        return loss, output

    def init_optimizer_fn(params: Params) -> Tuple[OptimizerState, UpdateFn]:
        #opt_init, opt_update, get_params = optimizers.momentum(step_size=lambda x: 0.001, mass=0.5)
        #opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 0.00002, b1=0.5)
        opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 0.000001)


        @jax.jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: Array) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = jax.value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)

            # grads = (grads[0], grads[1], jax.tree_map(lambda x: x * .1, grads[2]))

            opt_state = opt_update(i, grads, opt_state)

            return opt_state, loss, outputs

        opt_state = opt_init(params)

        return opt_state, update

    # evaluation
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.size == 1 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

    def close_value(value: Array) -> Array:
        return value if value.size == 1 else (jnp.mean(value) if value.ndim == 1 else img_decode_fn(value))

    def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
        to_cpu = jax.partial(jax.device_put, device=jax.devices('cpu')[0])
        new_output = jax.tree_map(to_cpu, new_output)
        return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)

    def log_output(output: Any) -> Any:
        return jax.tree_map(close_value, output)

    return init_fn, apply_fn, init_optimizer_fn, accumulate_output, log_output
