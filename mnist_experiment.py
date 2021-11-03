import shutil
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict, Tuple, FrozenSet

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental.stax import Conv, ConvTranspose, Dense, LeakyRelu, Relu, serial, Sigmoid
from more_itertools import powerset

from components.stax_extension import Array, PixelNorm2D, PRNGKey, Reshape, Shape, StaxLayer
from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.model import classifier_wrapper, model_wrapper
from model.train import Params, train
from test_bed import build_functions, cycle_transform_test, perform_tests, permute_transform_test, repeat_transform_test


def dummy():
    def init_fun(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return inputs

    return init_fun, apply_fun


Norm2D = PixelNorm2D
Norm1D = dummy()


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
        output = dec_apply_fn(params[1], new_latent_code)
        return output, exogenous_noise

    return init_fn, apply_fn


layers = (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='VALID'), Relu,
          Conv(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='VALID'), Relu,
          Conv(64 * 3, filter_shape=(4, 4), strides=(2, 2), padding='VALID'), Relu)
classifier_layers, disc_layers = layers, layers


def compile_fn(fn, params):
    def _fn(*args, **kwargs):
        return fn(params, *args, **kwargs)

    return _fn


if __name__ == '__main__':

    overwrite = False
    job_dir = Path('/tmp/class_test')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)

    noise_dim = 32
    train_datasets, test_dataset, parent_dims, marginals, input_shape = create_confounded_mnist_dataset()

    classifiers = {}
    for parent_name, parent_dim in parent_dims.items():
        model = classifier_wrapper(parent_name, parent_dim, classifier_layers)
        model_path = job_dir / parent_name / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            batch_size = 1024
            target_dist = frozenset((parent_name,))
            train_data = tfds.as_numpy(
                train_datasets[target_dist].batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
            test_data = tfds.as_numpy(test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
            params = train(model=model, input_shape=input_shape, job_dir=job_dir / parent_name, num_steps=1000,
                           train_data=train_data, test_data=test_data, log_every=1, eval_every=100, save_every=100)
        classifiers[parent_name] = compile_fn(model[1], params)

    interventions = (('color',), ('digit',))
    mode = 1
    parent_names = parent_dims.keys()
    for intervention in interventions:
        source_dist = frozenset(parent_names) if mode == 0 else frozenset()
        target_dist = frozenset(parent_names) if mode == 0 else frozenset(intervention)

        train_data = tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                          target_dist: train_datasets[target_dist]})
        test_data = tf.data.Dataset.zip({source_dist: test_dataset,
                                         target_dist: test_dataset})
        batch_size = 512
        train_data = tfds.as_numpy(train_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
        test_data = tfds.as_numpy(test_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))

        parent_name = intervention[0]
        mech = mechanism(parent_name, parent_dims, noise_dim)
        critic = serial(condition_on_parents(parent_dims), *disc_layers)
        model, _ = model_wrapper(source_dist, parent_name, marginals[parent_name], classifiers, critic, mech, noise_dim)

        params = train(model=model,
                       input_shape=input_shape,
                       job_dir=job_dir,
                       num_steps=3000,
                       train_data=train_data,
                       test_data=test_data,
                       log_every=1,
                       eval_every=500,
                       save_every=500)

    # model_path = job_dir / 'model.npy'
    # if model_path.exists():
    #     params = np.load(str(model_path), allow_pickle=True)
    # else:
    #     params = train(model=model,
    #                    input_shape=input_shape,
    #                    job_dir=job_dir,
    #                    num_steps=3000,
    #                    train_data=train_data,
    #                    test_data=test_data,
    #                    log_every=1,
    #                    eval_every=500,
    #                    save_every=500)
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
