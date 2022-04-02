from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import jit, tree_map, value_and_grad, vmap
from jax.example_libraries.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.example_libraries.stax import serial, Tanh

from components import Array, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.f_gan import f_gan
from datasets.utils import MarginalDistribution
from models.utils import concat_parents

# [[[image, parents]], [score, output]]
ClassifierFn = Callable[[Tuple[Array, Array]], Tuple[Array, Any]]
# [[image, parents, do_parents], do_image]
MechanismFn = Callable[[KeyArray, Array, Dict[str, Array], Dict[str, Array]], Array]


def l2(x: Array) -> Array:
    return vmap(lambda arr: jnp.linalg.norm(jnp.ravel(arr), ord=2))(x)


def mechanism(do_parent_names: Sequence[str],
              parent_dims: Dict[str, int],
              encoder_layers: Sequence[StaxLayer],
              decoder_layers: Sequence[StaxLayer]) -> StaxLayer:
    """
    Implements a
    Expects input to be in the range of -1 to 1
    Behaves like a partial mechanism if the set of do_parent_names is smaller than parent_dims.keys()

    :param do_parent_names:
    :param parent_dims:
    :param encoder_layers:
    :param decoder_layers:
    :return:
    """
    assert all([do_parent_name in parent_dims.keys() for do_parent_name in do_parent_names])
    enc_init_fn, enc_apply_fn = serial(*encoder_layers, Tanh)
    dec_init_fn, dec_apply_fn = serial(*decoder_layers)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        extra_dim = 2 * sum([value for key, value in parent_dims.items() if key in do_parent_names])
        enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + extra_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
        enc_params, dec_params = params
        _parents = concat_parents(parents)
        _do_parents = concat_parents(do_parents)
        latent_code = jnp.concatenate([enc_apply_fn(enc_params, image), *_parents, *_do_parents], axis=-1)
        return dec_apply_fn(dec_params, latent_code)

    return init_fn, apply_fn


# If do_parent_name =='all' uses full mechanism else uses partial mechanism
def functional_counterfactual(do_parent_name: str,
                              parent_dims: Dict[str, int],
                              classifiers: Dict[str, ClassifierFn],
                              critic_layers: Sequence[StaxLayer],
                              marginal_dists: Dict[str, MarginalDistribution],
                              mechanism_encoder_layers: Sequence[StaxLayer],
                              mechanism_decoder_layers: Sequence[StaxLayer],
                              is_invertible: Dict[str, bool],
                              constraint_function_power: int = 1,
                              from_joint: bool = True) -> Tuple[Model, Callable[[Params], MechanismFn]]:
    assert len(parent_dims) > 0
    assert do_parent_name in ['all', *parent_dims.keys()]
    assert parent_dims.keys() == classifiers.keys() == is_invertible.keys() == marginal_dists.keys()
    assert constraint_function_power >= 1
    do_parent_names = tuple(parent_dims.keys()) if do_parent_name == 'all' else (do_parent_name,)
    source_dist = frozenset() if from_joint else frozenset(parent_dims.keys())
    target_dist = frozenset(do_parent_names) if from_joint else frozenset(parent_dims.keys())
    divergence_init_fn, divergence_apply_fn = f_gan(critic=serial(critic_layers), mode='gan', trick_g=True)
    mechanism_init_fn, mechanism_apply_fn = mechanism(do_parent_names, parent_dims, mechanism_encoder_layers,
                                                      mechanism_decoder_layers)
    _is_invertible = all([is_invertible[parent_name] for parent_name in do_parent_names])

    def sampling_fn(rng: KeyArray, sample_shape: Shape, parents: Dict[str, Array]) -> Tuple[Array, Optional[Array]]:
        new_parents = {p_name: marginal_dists[p_name].sample(rng, sample_shape) for p_name in do_parent_names}
        do_parents = {**parents, **new_parents}
        order = ... if do_parent_name == 'all' else jnp.argsort(jnp.argmax(do_parents[do_parent_name], axis=-1))
        return do_parents, order

    def init_fn(rng: KeyArray, input_shape: Shape) -> Params:
        f_div_output_shape, f_div_params = divergence_init_fn(rng, input_shape)
        mechanism_output_shape, mechanism_params = mechanism_init_fn(rng, input_shape)
        return mechanism_output_shape, (f_div_params, mechanism_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Any]:
        divergence_params, mechanism_params = params
        (image, parents) = inputs[source_dist]

        # sample new parent(s) and perform functional counterfactual
        do_parents, order = sampling_fn(rng, (image.shape[0],), parents)
        do_image = mechanism_apply_fn(mechanism_params, image, parents, do_parents)

        # effectiveness constraint
        loss, output = divergence_apply_fn(divergence_params, inputs[target_dist], (do_image, do_parents))
        for parent_name, classifier in classifiers.items():
            cross_entropy, output[parent_name] = classifier((do_image, do_parents[parent_name]))
            loss = loss + cross_entropy
        output.update({'image': image[order], 'do_image': do_image[order]})

        # composition constraint
        image_null_intervention = image
        for i in range(1, constraint_function_power + 1):
            image_null_intervention = mechanism_apply_fn(mechanism_params, image_null_intervention, parents, parents)
            composition_constraint = jnp.mean(l2(image - image_null_intervention))
            loss = loss + composition_constraint
            output.update({f'image_null_intervention_{i:d}': image_null_intervention[order],
                           f'composition_constraint_{i:d}': composition_constraint})

        # reversibility constraint
        if _is_invertible:
            image_cycle: Optional[Array] = None
            for i in range(1, constraint_function_power + 1):
                if image_cycle is None:
                    image_forward = do_image
                else:
                    image_forward = mechanism_apply_fn(mechanism_params, image_cycle, parents, do_parents)
                image_cycle = mechanism_apply_fn(mechanism_params, image_forward, do_parents, parents)
                reversibility_constraint = jnp.mean(l2(image - image_cycle))
                loss = loss + reversibility_constraint
                output.update({f'image_cycle_{i:d}': image_cycle,
                               f'reversibility_constraint_{i:d}': reversibility_constraint})

        return loss, {'loss': loss[jnp.newaxis], **output}

    def init_optimizer_fn(params: Params, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
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

    def get_mechanism_fn(params: Params) -> MechanismFn:
        def mechanism_fn(rng: KeyArray, image: Array, parents: Dict[str, Array], do_parents: Dict[str, Array]) -> Array:
            return mechanism_apply_fn(params[1], image, parents, do_parents)

        return mechanism_fn

    return (init_fn, apply_fn, init_optimizer_fn), get_mechanism_fn
