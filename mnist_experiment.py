import argparse
from itertools import product
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import tensorflow as tf
from jax.experimental import optimizers
from jax.experimental.stax import Conv, ConvTranspose, Dense, Flatten, LeakyRelu, serial, Tanh

from components import Array, KeyArray, Params, Shape, StaxLayer
from components.stax_extension import PixelNorm2D, ResBlock, Reshape
from datasets.confounded_mnist import digit_colour_scenario, digit_fracture_colour_scenario
from run_experiment import ClassifierConfig, MechanismConfig, run_experiment, train_classifier

# import matplotlib
# matplotlib.use('tkagg')
tf.config.experimental.set_visible_devices([], 'GPU')


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

    def apply_fn(params: Params, image: Array, parent: Array, do_parent: Array) -> Array:
        enc_params, dec_params = params
        latent_code = jnp.concatenate([enc_apply_fn(enc_params, image), parent, do_parent], axis=-1)
        return dec_apply_fn(dec_params, latent_code)

    return init_fn, apply_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--scenario-name', dest='scenario_name', type=str, help='Name of scenario to run.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')
    args = parser.parse_args()

    scenarios = {'digit_colour_scenario': digit_colour_scenario,
                 'digit_fracture_colour_scenario': digit_fracture_colour_scenario}

    parameter_space = {'confound': [True, False], 'de_confound': [True, False], 'constraint_function_exponent': [1, 3]}

    layers = (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
              ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
              ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
              Flatten(), Dense(128), LeakyRelu)

    schedule = optimizers.piecewise_constant(boundaries=[2000, 4000], values=[1e-4, 1e-4 / 2, 1e-4 / 8])
    mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)

    scenario_dir = args.job_dir / args.scenario_name
    scenario_fn = scenarios[args.scenario_name]

    train_datasets, test_dataset, parent_dims, _, marginals, input_shape = scenario_fn(args.data_dir, False, False)

    classifier_configs = {parent_name: ClassifierConfig(parent_name=parent_name,
                                                        parent_dims=parent_dims,
                                                        input_shape=input_shape,
                                                        layers=layers,
                                                        optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                                                        batch_size=1024,
                                                        num_steps=2000,
                                                        log_every=1,
                                                        eval_every=50,
                                                        save_every=50) for parent_name in parent_dims.keys()}

    pseudo_oracles = {parent_name: train_classifier(job_dir=scenario_dir / 'pseudo_oracles',
                                                    seed=99,
                                                    train_datasets=train_datasets,
                                                    test_dataset=test_dataset,
                                                    config=config)
                      for parent_name, config in classifier_configs.items()}

    for confound, de_confound, constraint_function_exponent in product(*parameter_space.values()):
        if not confound and de_confound:
            continue
        job_name = f'confound_{str(confound)}_de_confounded_{str(de_confound)}_M_{constraint_function_exponent:d}'
        job_dir = scenario_dir / job_name
        train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape \
            = scenario_fn(args.data_dir, confound, de_confound)

        mechanism_configs = {parent_name: MechanismConfig(parent_name=parent_name,
                                                          parent_dims=parent_dims,
                                                          input_shape=input_shape,
                                                          critic_layers=layers,
                                                          mechanism=mechanism(parent_dims[parent_name]),
                                                          marginal_dist=marginals[parent_name],
                                                          is_invertible=is_invertible[parent_name],
                                                          condition_divergence_on_parents=True,
                                                          constraint_function_exponent=constraint_function_exponent,
                                                          optimizer=mechanism_optimizer,
                                                          batch_size=512,
                                                          num_steps=5000,
                                                          log_every=1,
                                                          eval_every=250,
                                                          save_every=250)
                             for parent_name, parent_dim in parent_dims.items()}

        run_experiment(job_dir,
                       args.overwrite,
                       args.seeds,
                       train_datasets,
                       test_dataset,
                       parent_dims,
                       classifier_configs,
                       mechanism_configs,
                       pseudo_oracles)
