import argparse
import pickle
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import optimizers
from jax.experimental.stax import Conv, ConvTranspose, Dense, Flatten, LeakyRelu, Tanh, serial

from components import Array, KeyArray, Params, Shape, StaxLayer
from components.stax_extension import PixelNorm2D, Reshape
from datasets.confounded_mnist import create_confounded_mnist_dataset, function_dict_to_confounding_fn, \
    get_colorize_fn, get_fracture_fn, get_thickening_fn, get_thinning_fn
from datasets.utils import ConfoundingFn, get_diagonal_confusion_matrix, get_uniform_confusion_matrix
from identifiability_tests import perform_tests, print_test_results
from models.functional_counterfactual import get_sampling_fn
from run_experiment import train_classifier, train_mechanism

# import matplotlib
# matplotlib.use('tkagg')
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
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')
    args = parser.parse_args()

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
        data_dir = str(args.data_dir / job_name)
        train_confounding_fns, test_confounding_fns, parent_dims, is_invertible = experiments[exp_name](control)
        train_datasets, test_dataset, marginals, input_shape = \
            create_confounded_mnist_dataset(data_dir, train_confounding_fns, test_confounding_fns, parent_dims)

        for seed in args.seeds:
            job_dir = args.job_dir / job_name / f'seed_{seed:d}'
            if job_dir.exists() and (job_dir / 'results.pickle').exists():
                if args.overwrite:
                    shutil.rmtree(job_dir)
                # else:
                #     continue

            classifiers = {}
            for parent_name, parent_dim in parent_dims.items():
                classifiers[parent_name] = train_classifier(job_dir=job_dir,
                                                            seed=seed,
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
            sampling_fns = {parent_name: get_sampling_fn(parent_dims[parent_name], False, marginals[parent_name])
                            for parent_name in parent_dims.keys()}
            for parent_name, parent_dim in parent_dims.items():
                mechanisms[parent_name] = train_mechanism(job_dir=job_dir,
                                                          seed=seed,
                                                          parent_name=parent_name,
                                                          parent_dims=parent_dims,
                                                          classifiers=classifiers,
                                                          critic_layers=layers,
                                                          mechanism=mechanism(parent_dim),
                                                          sampling_fn=sampling_fns[parent_name],
                                                          is_invertible=is_invertible[parent_name],
                                                          train_datasets=train_datasets,
                                                          test_dataset=test_dataset,
                                                          input_shape=input_shape,
                                                          optimizer=mechanism_optimizer,
                                                          batch_size=512,
                                                          num_steps=5000)

            # identifiability tests
            test_results = perform_tests(mechanisms, is_invertible, sampling_fns, classifiers, test_dataset)
            with open(job_dir / 'results.pickle', mode='wb') as f:
                pickle.dump(test_results, f)

        list_of_test_results = []
        for subdir in (args.job_dir / job_name).iterdir():
            with open(subdir / 'results.pickle', mode='rb') as f:
                list_of_test_results.append(pickle.load(f))
        print_test_results(list_of_test_results)
