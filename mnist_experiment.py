import shutil
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
from jax.experimental.stax import Conv, Dense, Flatten, Relu, serial, Sigmoid

from components.stax_layers import layer_norm
from components.typing import Array, PRNGKey, Shape, StaxLayer
from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.model import build_model
from model.train import Params, train

LayerNorm2D = layer_norm(axis=(1, 2, 3))
LayerNorm1D = layer_norm(axis=(1,))


def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int) -> StaxLayer:
    hidden_dim = 512
    assert hidden_dim > noise_dim

    # This one works for color at least
    enc_init_fn, enc_apply_fn = serial(Conv(32, filter_shape=(3, 3), padding='SAME'), LayerNorm2D, Relu,
                                       Conv(64, filter_shape=(3, 3), padding='SAME'), LayerNorm2D, Relu,
                                       Conv(128, filter_shape=(3, 3), padding='SAME'), LayerNorm2D, Relu,
                                       Conv(64, filter_shape=(3, 3), padding='SAME'), LayerNorm2D, Relu,
                                       Conv(32, filter_shape=(3, 3), padding='SAME'), LayerNorm2D, Relu,
                                       Conv(3, filter_shape=(3, 3), padding='SAME'), Sigmoid)

    extra_dim = sum(parent_dims.values()) + parent_dims[parent_name]

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        enc_output_shape, enc_params = enc_init_fn(rng, (*input_shape[:-1], input_shape[-1] + extra_dim))
        return enc_output_shape, (enc_params,)

    def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array, noise: Array) \
            -> Tuple[Array, Array]:
        s = (*inputs.shape[:-1], extra_dim)
        __in = jnp.concatenate((*[parents[key] for key in parent_dims.keys()], do_parent), axis=-1)
        _in = jnp.concatenate((inputs, broadcast(__in, s)), axis=-1)
        output = enc_apply_fn(params[0], _in)
        return output, noise

    return init_fn, apply_fn


# def mechanism(parent_name: str, parent_dims: Dict[str, int], noise_dim: int) -> StaxLayer:
#     hidden_dim = 512
#     assert hidden_dim > noise_dim
#     enc_init_fn, enc_apply_fn = \
#         serial(Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LayerNorm2D, LeakyRelu,
#                Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LayerNorm2D, LeakyRelu,
#                Reshape((-1, 7 * 7 * 128)), Dense(hidden_dim), Sigmoid)
#
#     dec_init_fn, dec_apply_fn = \
#         serial(Dense(1024), LayerNorm1D, Relu,
#                Dense(7 * 7 * 128), LayerNorm1D, Relu, Reshape((-1, 7, 7, 128)),
#                ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LayerNorm2D, Relu,
#                ConvTranspose(3, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Sigmoid)
#
#     extra_dim = sum(parent_dims.values()) + parent_dims[parent_name]
#
#     def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
#         enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
#         output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + extra_dim))
#         return output_shape, (enc_params, dec_params)
#
#     def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array, noise: Array) \
#             -> Tuple[Array, Array]:
#         latent_code = enc_apply_fn(params[0], inputs)
#         exogenous_noise = latent_code[..., :noise_dim]
#         new_latent_code = jnp.concatenate([noise, latent_code[..., noise_dim:], *parents.values(), do_parent], axis=-1)
#         new_latent_code = jnp.concatenate([latent_code, *parents.values(), do_parent], axis=-1)
#
#         output = dec_apply_fn(params[1], new_latent_code)
#         return output, exogenous_noise
#
#     return init_fn, apply_fn


layers = (Conv(32, filter_shape=(3, 3), padding='VALID'), LayerNorm2D, Relu,
          Conv(16, filter_shape=(3, 3), padding='VALID'), LayerNorm2D, Relu,
          Conv(16, filter_shape=(3, 3), padding='VALID'), LayerNorm2D, Relu,
          Conv(8, filter_shape=(3, 3), padding='VALID'), LayerNorm2D, Relu,
          Flatten, Dense(512), LayerNorm1D, Relu,
          Flatten, Dense(512), LayerNorm1D, Relu)

if __name__ == '__main__':

    overwrite = True
    job_dir = Path('/tmp/grand_canyon_cycle_0')
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)
    if job_dir.exists() and not overwrite:
        exit(1)

    noise_dim = 8
    train_data, test_data, parent_dims, marginals, input_shape = \
        create_confounded_mnist_dataset(batch_size=1024, debug=True)
    print('Loaded dataset...')

    interventions = (('color',), ('thickness',))

    # interventions = (('thickness',),)

    model = build_model(parent_dims, marginals, interventions, layers, mechanism, noise_dim, mode=0)
    train(model=model,
          input_shape=input_shape,
          job_dir=job_dir,
          num_steps=10000,
          train_data=train_data,
          test_data=test_data,
          log_every=1,
          eval_every=50,
          save_every=100)
