import argparse
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('tkagg')
from jax.experimental import optimizers
from jax.experimental.stax import Conv, ConvTranspose, LeakyRelu, Tanh
from jax.experimental.stax import Dense, Flatten, serial

from components import Array, KeyArray, Params, Shape, StaxLayer
from components.stax_extension import PixelNorm2D, Reshape
from datasets.confounded_mnist import create_confounded_mnist_dataset, function_dict_to_confounding_fn, \
    get_colorize_fn, get_fracture_fn, get_thickening_fn, get_thinning_fn
from datasets.utils import ConfoundingFn, get_diagonal_confusion_matrix, get_uniform_confusion_matrix
from models.functional_counterfactual import get_sampling_fn
from run_experiment import train_classifier, train_mechanism

tf.config.experimental.set_visible_devices([], 'GPU')

Experiment = Tuple[List[ConfoundingFn], List[ConfoundingFn], Dict[str, int], Dict[str, bool]]


def mechanism(parent_dim: int) -> StaxLayer:
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

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + 2 * parent_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, inputs: Array, parent: Array, do_parent: Array) -> Array:
        enc_params, dec_params = params
        latent_code = jnp.concatenate([enc_apply_fn(enc_params, inputs), parent, do_parent], axis=-1)
        return dec_apply_fn(dec_params, latent_code)

    return init_fn, apply_fn


def ResBlock(out_features: int, filter_shape: Tuple[int, int], strides: Tuple[int, int]) -> StaxLayer:
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


def experiment_0(control: bool = False) -> Experiment:
    parent_dims = {'digit': 10, 'color': 10}
    is_invertible = {'digit': False, 'color': True}
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1) if not control else test_colorize_cm
    train_colorize_fn = get_colorize_fn(train_colorize_cm)
    test_colorize_fn = get_colorize_fn(test_colorize_cm)
    return [train_colorize_fn], [test_colorize_fn], parent_dims, is_invertible


# Even digits have much higher chance of thick
def experiment_1(control: bool = False) -> Experiment:
    parent_dims = {'digit': 10, 'thickness': 2, 'color': 10}
    is_invertible = {'digit': False, 'thickness': True, 'color': True}

    even_heavy_cm = np.zeros(shape=(10, 2))
    even_heavy_cm[0:-1:2] = (.1, .9)
    even_heavy_cm[1::2] = (.9, .1)

    test_thickening_cm = get_uniform_confusion_matrix(10, 2)
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    train_thickening_cm = even_heavy_cm if not control else test_thickening_cm
    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1) if not control else test_colorize_cm

    function_dict = {0: get_thinning_fn(), 1: get_thickening_fn()}
    train_thickening_fn = function_dict_to_confounding_fn(function_dict, train_thickening_cm)
    test_thickening_fn = function_dict_to_confounding_fn(function_dict, test_thickening_cm)
    train_colorize_fn = get_colorize_fn(train_colorize_cm)
    test_colorize_fn = get_colorize_fn(test_colorize_cm)
    return [train_thickening_fn, train_colorize_fn], [test_thickening_fn, test_colorize_fn], parent_dims, is_invertible


def experiment_2(control: bool = False) -> Experiment:
    parent_dims = {'digit': 10, 'fracture': 2, 'color': 10}
    is_invertible = {'digit': False, 'fracture': False, 'color': True}

    even_heavy_cm = np.zeros(shape=(10, 2))
    even_heavy_cm[0:-1:2] = (.1, .9)
    even_heavy_cm[1::2] = (.9, .1)

    test_thickening_cm = get_uniform_confusion_matrix(10, 2)
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    train_thickening_cm = even_heavy_cm if not control else test_thickening_cm
    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1) if not control else test_colorize_cm

    function_dict = {0: lambda x: x, 1: get_fracture_fn(num_frac=1)}
    train_thickening_fn = function_dict_to_confounding_fn(function_dict, train_thickening_cm)
    test_thickening_fn = function_dict_to_confounding_fn(function_dict, test_thickening_cm)
    train_colorize_fn = get_colorize_fn(train_colorize_cm)
    test_colorize_fn = get_colorize_fn(test_colorize_cm)
    return [train_thickening_fn, train_colorize_fn], [test_thickening_fn, test_colorize_fn], parent_dims, is_invertible


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seed', dest='seed', type=int, help='random seed')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    experiments = {'exp_0': experiment_0, 'exp_1': experiment_1, 'exp_2': experiment_2}

    layers: Tuple[StaxLayer, ...] = (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
                                     ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
                                     ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
                                     Flatten, Dense(128), LeakyRelu)

    schedule = optimizers.piecewise_constant(boundaries=[2000, 4000], values=[5e-4, 5e-4 / 2, 5e-4 / 8])
    mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)

    for exp_name, control in product(experiments, [True, False]):
        job_name = exp_name + ('_control' if control else '')
        job_dir = args.job_dir / job_name
        job_dir = Path(job_dir)
        if job_dir.exists() and args.overwrite:
            shutil.rmtree(job_dir)
        data_dir = str(args.data_dir / job_name)

        train_confounding_fns, test_confounding_fns, parent_dims, is_invertible = experiments[exp_name](control)
        train_datasets, test_dataset, marginals, input_shape = \
            create_confounded_mnist_dataset(data_dir, train_confounding_fns, test_confounding_fns, parent_dims)

        classifiers = {}
        for parent_name, parent_dim in parent_dims.items():
            classifiers[parent_name] = train_classifier(job_dir=job_dir,
                                                        seed=args.seed,
                                                        parent_name=parent_name,
                                                        num_classes=parent_dim,
                                                        layers=layers,
                                                        train_datasets=train_datasets,
                                                        test_dataset=test_dataset,
                                                        input_shape=input_shape,
                                                        optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                                                        batch_size=1024,
                                                        num_steps=2000)
        mechanisms = {}
        for parent_name, parent_dim in parent_dims.items():
            sampling_fn = get_sampling_fn(parent_dims[parent_name], False, marginals[parent_name])
            mechanisms[parent_name] = train_mechanism(job_dir=job_dir,
                                                      seed=args.seed,
                                                      parent_name=parent_name,
                                                      parent_dims=parent_dims,
                                                      classifiers=classifiers,
                                                      critic_layers=layers,
                                                      mechanism=mechanism(parent_dim),
                                                      sampling_fn=sampling_fn,
                                                      is_invertible=is_invertible[parent_name],
                                                      train_datasets=train_datasets,
                                                      test_dataset=test_dataset,
                                                      input_shape=input_shape,
                                                      optimizer=mechanism_optimizer,
                                                      batch_size=512,
                                                      num_steps=5000)
            # Test
            # repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
            #                for p_name, mechanism in mechanisms.items()}
            # cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
            #               for p_name, mechanism in mechanisms.items()}
            # permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
            #                                                    for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
            # tests = {**repeat_test, **cycle_test, **permute_test}
            # res = perform_tests(test_data, tests)
