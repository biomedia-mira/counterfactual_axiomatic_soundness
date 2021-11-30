import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.stax import Conv, ConvTranspose, Dense, Flatten, LeakyRelu, Tanh, serial

from components.f_gan import f_gan
from components.stax_extension import Array, PRNGKey, Params, PixelNorm2D, Reshape, Shape, StaxLayer
from datasets.confounded_mnist import create_confounded_mnist_dataset


def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


def condition_on_parents(parent_dims: Dict[str, int]) -> StaxLayer:
    def init_fn(rng: PRNGKey, shape: Shape) -> Tuple[Shape, Params]:
        return (*shape[:-1], shape[-1] + sum(parent_dims.values())), ()

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Array:
        image, parents = inputs
        shape = (*image.shape[:-1], sum(parent_dims.values()))
        _parents = jnp.concatenate([parents[key] for key in parent_dims.keys()], axis=-1)
        return jnp.concatenate((image, broadcast(_parents, shape)), axis=-1)

    return init_fn, apply_fn


def ResBlock(out_features: int, filter_shape: Tuple[int, int], strides: Tuple[int, int]):
    _init_fn, _apply_fn = serial(Conv(out_features, filter_shape=(3, 3), strides=(1, 1), padding='SAME'),
                                 PixelNorm2D, LeakyRelu,
                                 Conv(out_features, filter_shape=filter_shape, strides=strides, padding='SAME'),
                                 PixelNorm2D, LeakyRelu)

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        output = _apply_fn(params, inputs)
        residual = jax.image.resize(jnp.repeat(inputs, output.shape[-1] // inputs.shape[-1], axis=-1),
                                    shape=output.shape, method='bilinear')
        return output + residual

    return _init_fn, apply_fn


layers = (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
          ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
          ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
          Flatten, Dense(128), LeakyRelu)

classifier_layers = layers


def mechanism(parent_dim: int, noise_dim: int) -> StaxLayer:
    hidden_dim = 1024
    enc_init_fn, enc_apply_fn = \
        serial(Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
               Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
               Reshape((-1, 7 * 7 * 128)), Dense(hidden_dim), LeakyRelu)

    dec_init_fn, dec_apply_fn = \
        serial(Dense(7 * 7 * 128), LeakyRelu, Reshape((-1, 7, 7, 128)),
               ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
               ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
               Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'), Tanh)

    extra_dim = 2 * parent_dim + noise_dim

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + extra_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, inputs: Array, parent: Array, do_parent: Array, exogenous_noise: Array) -> Array:
        enc_params, dec_params = params
        latent_code = jnp.concatenate([exogenous_noise, enc_apply_fn(enc_params, inputs), parent, do_parent], axis=-1)
        return dec_apply_fn(dec_params, latent_code)

    return init_fn, apply_fn


if __name__ == '__main__':

    overwrite = False
    job_dir = Path('/tmp/test_3')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)

    train_datasets, test_dataset, parent_dims, marginals, input_shape = create_confounded_mnist_dataset()

    noise_dim = 64
    f_gan = f_gan(serial(condition_on_parents(parent_dims), *layers, Flatten, Dense(1)), mode='gan', trick_g=True)

    abductor = serial(condition_on_parents(parent_dims),
                      Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
                      Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
                      Flatten, Dense(noise_dim))

    mechanism(parent_dims[parent_name], noise_dim)

    #
    # classifiers, divergences, mechanisms = build_functions(params, *functions)
    # repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
    #                for p_name, mechanism in mechanisms.items()}
    # cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
    #               for p_name, mechanism in mechanisms.items()}
    # permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
    #                                                    for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
    # tests = {**repeat_test, **cycle_test, **permute_test}
    # res = perform_tests(test_data, tests)
