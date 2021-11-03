import shutil
from pathlib import Path
from typing import Any
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental.stax import Conv, ConvTranspose, Dense, LeakyRelu, Relu, serial, Sigmoid, Tanh
from more_itertools import powerset

from components.stax_extension import layer_norm, LayerNorm2D, Reshape,Array, PRNGKey, Shape, StaxLayer
from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.model import build_model
from model.train import Params, train
from test_bed import build_functions, cycle_transform_test, perform_tests, permute_transform_test, repeat_transform_test
import tensorflow_datasets as tfds

import jax


def dummy():
    def init_fun(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return inputs

    return init_fun, apply_fun


Norm2D = LayerNorm2D
Norm1D = dummy()

#
# Norm2D = BatchNorm(axis=(0, 1, 2))
# Norm1D = BatchNorm(axis=(0,))


Norm2D = layer_norm(axis=(3,))
Norm1D = dummy()
PixelNorm = layer_norm(axis=(3,))


def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


# def synthesis_block(fmaps_out: int, c_dim: int, num_image_channels: int = 3) -> StaxLayer:
#     block_init_fn, block_apply_fn = serial(Conv(fmaps_out, filter_shape=(3, 3),padding='SAME'), PixelNorm, LeakyRelu,
#                                            ConvTranspose(fmaps_out, filter_shape=(4, 4), strides=(2, 2),
#                                                          padding='SAME'),
#                                            PixelNorm, LeakyRelu)
#
#     to_rgb_init_fn, to_rgb_apply_fn = Conv(num_image_channels, filter_shape=(1, 1))
#
#     def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
#         k1, k2 = jax.random.split(rng, num=2)
#         output_shape, block_params = block_init_fn(k1, (*input_shape[:-1], input_shape[-1] + c_dim))
#         _, to_rgb_params = to_rgb_init_fn(k2, output_shape)
#         return output_shape, (block_params, to_rgb_params)
#
#     def apply_fn(params: Params, inputs: Tuple[Array, Array, Array], **kwargs: Any) -> Tuple[
#         Array, Array, Array]:
#         block_params, to_rgb_params = params
#         x, y, c = inputs
#         c_low_res = jax.image.resize(c, (*x.shape[:-1], c.shape[-1]), method='bilinear')
#         x = block_apply_fn(block_params, jnp.concatenate((x, c_low_res), axis=-1))
#         y = jax.image.resize(y, (*x.shape[:-1], y.shape[-1]), method='bilinear')
#         y = to_rgb_apply_fn(to_rgb_params, x)[0] + y
#         return x, y, c
#
#     return init_fn, apply_fn
#
#
# def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int,
#               c_dim: int = 64,
#               num_image_channels: int = 3,
#               resolution: int = 32) -> StaxLayer:
#     resolution_log2 = int(jnp.log2(resolution))
#     extra_dim = 2 * parent_dims[parent_name] + noise_dim
#
#     def nf(res: int):
#         return 64
#
#     c_encode_init_fn, c_encode_apply_fb = serial(Conv(c_dim, filter_shape=(3, 3), padding='SAME'), PixelNorm, LeakyRelu,
#                                                  Conv(c_dim, filter_shape=(3, 3), padding='SAME'), PixelNorm, LeakyRelu)
#
#     # transform_init_fn, transform_apply_fn = \
#     #     serial(*[synthesis_block(nf(res - 1), c_dim, num_image_channels) for res in range(2, resolution_log2 + 1)])
#
#     transform_init_fn, transform_apply_fn = \
#         serial(*[synthesis_block(nf(res - 1), c_dim, num_image_channels) for res in range(2, resolution_log2)])
#
#     def mechanism_init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
#         k1, k2, k3 = jax.random.split(rng, num=3)
#         const_input = jax.random.normal(k1, (1, 4, 4, nf(1)))
#         _, c_encode_params = c_encode_init_fn(k2, (*input_shape[:-1], input_shape[-1] + extra_dim))
#         output_shape, transform_params = transform_init_fn(k2, const_input.shape)
#         return output_shape, (const_input, c_encode_params, transform_params)
#
#     def mechanism_apply_fn(params: Params,
#                            inputs: Array,
#                            parents: Dict[str, Array],
#                            do_parent: Array,
#                            do_noise: Array, **kwargs: Any) -> Array:
#         const_input, c_encode_params, transform_params = params
#         _parents = jnp.concatenate((parents[parent_name], do_parent, do_noise), axis=-1)
#         c = jnp.concatenate((inputs, broadcast(_parents, (*inputs.shape[:-1], _parents.shape[-1]))), axis=-1)
#         c = c_encode_apply_fb(c_encode_params, c)
#         x = jnp.repeat(const_input, c.shape[0], axis=0)
#         y = jnp.zeros((*x.shape[:-1], num_image_channels))
#         x, y, c = transform_apply_fn(transform_params, (x, y, c))
#         return y, do_noise
#
#     return mechanism_init_fn, mechanism_apply_fn


##
from jax.nn.initializers import he_normal, zeros
def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int) -> StaxLayer:
    hidden_dim = 1024
    assert hidden_dim > noise_dim
    enc_init_fn, enc_apply_fn = \
        serial(Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME', W_init=he_normal()), Norm2D, LeakyRelu,
               Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME', W_init=he_normal()), Norm2D, LeakyRelu,
               Reshape((-1, 7 * 7 * 128)), Dense(hidden_dim), LeakyRelu)

    dec_init_fn, dec_apply_fn = \
        serial(Dense(7 * 7 * 128), Norm1D, LeakyRelu, Reshape((-1, 7, 7, 128)),
               ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME', W_init=he_normal()), Norm2D, LeakyRelu,
               ConvTranspose(3, filter_shape=(4, 4), strides=(2, 2), padding='SAME', W_init=he_normal()))

    extra_dim = sum(parent_dims.values()) + parent_dims[parent_name]

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + extra_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array, noise: Array) \
            -> Tuple[Array, Array]:
        latent_code = enc_apply_fn(params[0], inputs)
        exogenous_noise = latent_code[..., :noise_dim]
        new_latent_code = jnp.concatenate(
            [noise, latent_code[..., noise_dim:], *[parents[key] for key in parent_dims.keys()], do_parent], axis=-1)
        # new_latent_code = jnp.concatenate([latent_code, *[parents[key] for key in parent_dims.keys()], do_parent], axis=-1)
        output = dec_apply_fn(params[1], new_latent_code)
        return output, exogenous_noise

    return init_fn, apply_fn


layers = (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=he_normal()), Relu,
          Conv(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=he_normal()), Relu,
          Conv(64 * 3, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=he_normal()), Relu)
classifier_layers, disc_layers = layers, layers

# def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int) -> StaxLayer:
#     # This one works for color at least
#     enc_init_fn, enc_apply_fn = serial(Conv(32, filter_shape=(3, 3), padding='SAME'), Norm2D, Relu,
#                                        Conv(32, filter_shape=(3, 3), padding='SAME'), Norm2D, Relu,
#                                        Conv(32, filter_shape=(3, 3), padding='SAME'), Norm2D, Relu,
#                                        Conv(3 + noise_dim, filter_shape=(3, 3), padding='SAME'), Sigmoid)
#
#     # extra_dim = sum(parent_dims.values()) + parent_dims[parent_name]
#
#     extra_dim = 2 * parent_dims[parent_name] + noise_dim
#
#     def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
#         enc_output_shape, enc_params = enc_init_fn(rng, (*input_shape[:-1], input_shape[-1] + extra_dim))
#         return enc_output_shape, (enc_params,)
#
#     def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array, do_noise: Array) \
#             -> Tuple[Array, Array]:
#         # __in = jnp.concatenate((*[parents[key] for key in parent_dims.keys()], do_parent), axis=-1)
#         __in = jnp.concatenate((parents[parent_name], do_parent, do_noise), axis=-1)
#         _in = jnp.concatenate((inputs, broadcast(__in, (*inputs.shape[:-1], extra_dim))), axis=-1)
#         output = enc_apply_fn(params[0], _in)
#         do_image, noise = output[..., :3], jnp.mean(output[..., 3:], axis=(1, 2))
#         return do_image, noise
#
#     return init_fn, apply_fn
#
#
# classifier_layers = (Conv(32, filter_shape=(3, 3), padding='VALID'), Relu,
#                      Conv(16, filter_shape=(3, 3), padding='VALID'), Relu,
#                      Conv(8, filter_shape=(3, 3), padding='VALID'), Relu)
# disc_layers = (Conv(32, filter_shape=(3, 3), padding='VALID'), Relu,
#                Flatten, Dense(512), Relu, Dense(512), Relu)

if __name__ == '__main__':

    overwrite = True
    job_dir = Path('/tmp/grand_canyon_cycle_0')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)

    noise_dim = 32
    train_data, test_data, parent_dims, marginals, input_shape = create_confounded_mnist_dataset()
    train_data = tf.data.Dataset.zip({key: train_data[key] for key in
                                      [frozenset(()), frozenset(('digit',)), frozenset(('color',)),
                                       frozenset(('thickness',))]})
    test_data = tf.data.Dataset.zip(
        {frozenset(parent_set): test_data for parent_set in powerset(parent_dims.keys())})
    batch_size = 512
    train_data = tfds.as_numpy(train_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    test_data = tfds.as_numpy(test_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    print('Loaded dataset...')

    interventions = (('digit',), ('color',))

    # interventions = (('color',), ('thickness',),)

    model, functions = build_model(parent_dims, marginals, interventions, classifier_layers, disc_layers, mechanism,
                                   noise_dim, mode=1)

    model_path = job_dir / 'model.npy'
    if model_path.exists():
        params = np.load(str(model_path), allow_pickle=True)
    else:
        params = train(model=model,
                       input_shape=input_shape,
                       job_dir=job_dir,
                       num_steps=3000,
                       train_data=train_data,
                       test_data=test_data,
                       log_every=1,
                       eval_every=500,
                       save_every=500)
    classifiers, divergences, mechanisms = build_functions(params, *functions)
    repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
                   for p_name, mechanism in mechanisms.items()}
    cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
                  for p_name, mechanism in mechanisms.items()}
    permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
                                                       for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
    tests = {**repeat_test, **cycle_test, **permute_test}
    res = perform_tests(test_data, tests)
