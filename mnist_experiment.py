import shutil
from pathlib import Path
from typing import Any
from typing import Dict, Tuple, Callable

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental.stax import Conv, ConvTranspose, Dense, LeakyRelu, serial, Flatten, Tanh

from components.stax_extension import Array, PixelNorm2D, PRNGKey, Reshape, Shape, StaxLayer
from datasets.mnist_mechanisms import create_confounded_mnist_dataset
from model.model import classifier_wrapper, model_wrapper
from model.train import Params, train
import jax
from datasets.augmentation import random_crop_and_rescale

Norm2D = PixelNorm2D


def ResBlock(out_features: int, filter_shape: Tuple[int, int], strides: Tuple[int, int]):
    _init_fn, _apply_fn = serial(Conv(out_features, filter_shape=(3, 3), strides=(1, 1), padding='SAME'),
                                 Norm2D, LeakyRelu,
                                 Conv(out_features, filter_shape=filter_shape, strides=strides, padding='SAME'),
                                 Norm2D, LeakyRelu)

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        output = _apply_fn(params, inputs)
        residual = jax.image.resize(jnp.repeat(inputs, output.shape[-1] // inputs.shape[-1], axis=-1),
                                    shape=output.shape, method='bilinear')
        return output + residual

    return _init_fn, apply_fn


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
    enc_init_fn, enc_apply_fn = \
        serial(Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               Reshape((-1, 7 * 7 * 128)), Dense(hidden_dim), LeakyRelu)

    dec_init_fn, dec_apply_fn = \
        serial(Dense(7 * 7 * 128), LeakyRelu, Reshape((-1, 7, 7, 128)),
               ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Norm2D, LeakyRelu,
               Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'), Tanh)

    extra_dim = 2 * parent_dims[parent_name] + noise_dim

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + extra_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, inputs: Array, parents: Dict[str, Array], do_parent: Array, exogenous_noise: Array) \
            -> Tuple[Array, Array]:
        latent_code = jnp.concatenate(
            [exogenous_noise, enc_apply_fn(params[0], inputs), parents[parent_name], do_parent], axis=-1)
        return dec_apply_fn(params[1], latent_code)

    return init_fn, apply_fn


def abductor(parent_dims, noise_dim: int) -> StaxLayer:
    init_fn, apply_fn = \
        serial(condition_on_parents(parent_dims),
               Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
               Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
               Flatten, Dense(noise_dim))

    # stax_wrapper(lambda x: x / jnp.linalg.norm(x, ord=2)))
    return init_fn, apply_fn


# layers = (Conv(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
#           Conv(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
#           Conv(64 * 3, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
#           Flatten, Dense(1024), LeakyRelu)

layers = (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
          ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
          ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
          Flatten, Dense(128), LeakyRelu)

classifier_layers = layers
disc_layers = (*layers, Flatten, Dense(1))


def compile_fn(fn: Callable, params: Params) -> Callable:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return fn(params, *args, **kwargs)

    return _fn


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))


def select_parent(parent_name: str) -> Callable[[tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, tf.Tensor]]:
    def _select(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        return image, parents[parent_name]

    return _select


if __name__ == '__main__':

    overwrite = False
    job_dir = Path('/tmp/test_3')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)

    train_datasets, test_dataset, parent_dims, marginals, input_shape = create_confounded_mnist_dataset()

    # Train classifiers
    classifiers = {}
    batch_size = 1024
    for parent_name, parent_dim in parent_dims.items():
        classifier_model = classifier_wrapper(parent_dim, classifier_layers)
        model_path = job_dir / parent_name / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            target_dist = frozenset((parent_name,))
            train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent(parent_name)), batch_size)
            test_data = to_numpy_iterator(test_dataset.map(select_parent(parent_name)), batch_size)
            params = train(model=classifier_model,
                           input_shape=input_shape,
                           job_dir=job_dir / parent_name,
                           num_steps=2000,
                           train_data=train_data,
                           test_data=test_data,
                           log_every=1,
                           eval_every=100,
                           save_every=100)
        classifiers[parent_name] = compile_fn(fn=classifier_model[1], params=params)

    # Train counterfactual functions
    interventions = (('digit',), ('color',))
    noise_dim = 64
    batch_size = 512
    mode = 1
    parent_names = parent_dims.keys()
    for intervention in interventions:
        source_dist = frozenset(parent_names) if mode == 0 else frozenset()
        target_dist = frozenset(parent_names) if mode == 0 else frozenset(intervention)
        train_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                                            target_dist: train_datasets[target_dist]}), batch_size)
        test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset,
                                                           target_dist: test_dataset}), batch_size)

        parent_name = intervention[0]
        mech = mechanism(parent_name, parent_dims, noise_dim)
        critic = serial(condition_on_parents(parent_dims), *disc_layers)
        model, _ = model_wrapper(source_dist, parent_name, marginals[parent_name], classifiers, critic, mech,
                                 abductor(parent_dims, noise_dim))

        params = train(model=model,
                       input_shape=input_shape,
                       job_dir=job_dir,
                       num_steps=5000,
                       train_data=train_data,
                       test_data=test_data,
                       log_every=1,
                       eval_every=500,
                       save_every=500)

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
