import shutil
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax.experimental.stax import Conv, Dense, Flatten, Relu, serial, Sigmoid, LeakyRelu, ConvTranspose, BatchNorm

from components.stax_layers import LayerNorm1D, LayerNorm2D, Reshape, layer_norm, residual
from components.typing import Array, PRNGKey, Shape, StaxLayer
from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.model import build_model
from model.train import Params, train
from test_bed import build_functions, perform_tests, repeat_transform_test, permute_transform_test, cycle_transform_test
from typing import Any


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


def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int) -> StaxLayer:
    hidden_dim = 1024
    assert hidden_dim > noise_dim
    enc_init_fn, enc_apply_fn = \
        serial(Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               Reshape((-1, 7 * 7 * 128)), Dense(hidden_dim), LeakyRelu)

    dec_init_fn, dec_apply_fn = \
        serial(Dense(7 * 7 * 128), Norm1D, LeakyRelu, Reshape((-1, 7, 7, 128)),
               ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               ConvTranspose(3, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Sigmoid)

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


layers = (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='VALID'), Relu,
          Conv(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='VALID'), Relu,
          Conv(64 * 3, filter_shape=(4, 4), strides=(2, 2), padding='VALID'), Relu)
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
disc_layers = classifier_layers
if __name__ == '__main__':

    overwrite = True
    job_dir = Path('/tmp/grand_canyon_cycle_0')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)

    noise_dim = 32
    train_data, test_data, parent_dims, marginals, input_shape = \
        create_confounded_mnist_dataset(batch_size=1024, debug=False)
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
