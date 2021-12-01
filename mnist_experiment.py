import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optimizers
from jax.experimental.stax import Conv, ConvTranspose, Dense, Flatten, LeakyRelu, Tanh, serial

from components import Array, KeyArray, Params, PixelNorm2D, Reshape, Shape, StaxLayer
from datasets.confounded_mnist import create_confounded_mnist_dataset, function_dict_to_confounding_fn, \
    get_colorize_fn, get_thickening_fn, get_thinning_fn
from datasets.utils import ConfoundingFn, get_diagonal_confusion_matrix, get_uniform_confusion_matrix
from run_experiment import run_experiment


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

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        enc_output_shape, enc_params = enc_init_fn(rng, input_shape)
        output_shape, dec_params = dec_init_fn(rng, (*enc_output_shape[:-1], enc_output_shape[-1] + extra_dim))
        return output_shape, (enc_params, dec_params)

    def apply_fn(params: Params, inputs: Array, parent: Array, do_parent: Array, exogenous_noise: Array) -> Array:
        enc_params, dec_params = params
        latent_code = jnp.concatenate([exogenous_noise, enc_apply_fn(enc_params, inputs), parent, do_parent], axis=-1)
        return dec_apply_fn(dec_params, latent_code)

    return init_fn, apply_fn


def experiment_0(control: bool = False) -> Tuple[List[ConfoundingFn], List[ConfoundingFn], Dict[str, int]]:
    parent_dims = {'digit': 10, 'color': 10}
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1) if not control else test_colorize_cm
    train_colorize_fn = get_colorize_fn(train_colorize_cm)
    test_colorize_fn = get_colorize_fn(test_colorize_cm)
    return [train_colorize_fn], [test_colorize_fn], parent_dims


# Even digits have much higher chance of swelling
def experiment_1(control: bool = False) -> Tuple[List[ConfoundingFn], List[ConfoundingFn], Dict[str, int]]:
    parent_dims = {'digit': 10, 'thickness': 2, 'color': 10}

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

    return [train_thickening_fn, train_colorize_fn], [test_thickening_fn, test_colorize_fn], parent_dims


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    args = parser.parse_args()

    noise_dim = 64
    layers = (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
              ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
              ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
              Flatten, Dense(128), LeakyRelu)
    classifier_layers = layers
    critic_layers = layers
    abductor_layers = (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
                       Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu)

    for control in [True, False]:
        for i, experiment in enumerate([experiment_0, experiment_1]):
            job_dir = args.job_dir / f'exp_{i:d}' + ('_control' if control else '')
            train_confounding_fns, test_confounding_fns, parent_dims = experiment(control)
            train_datasets, test_dataset, marginals, input_shape = \
                create_confounded_mnist_dataset('./data', train_confounding_fns, test_confounding_fns, parent_dims)
            run_experiment(job_dir=job_dir,
                           seed=1,
                           train_datasets=train_datasets,
                           test_dataset=test_dataset,
                           parent_dims=parent_dims,
                           marginals=marginals,
                           input_shape=input_shape,
                           classifier_layers=classifier_layers,
                           classifier_optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                           classifier_batch_size=1024,
                           classifier_num_steps=2000,
                           critic_layers=critic_layers,
                           abductor_layers=abductor_layers,
                           mechanism_constructor=mechanism,
                           noise_dim=noise_dim,
                           counterfactual_batch_size=512,
                           counterfactual_num_steps=5000,
                           overwrite=args.overwrite)
