import argparse
from pathlib import Path
from typing import cast, List

import tensorflow as tf
from jax.example_libraries import optimizers
from jax.example_libraries.stax import Conv, Dense, FanInConcat, FanOut, Flatten, LeakyRelu, parallel, serial, Tanh

from components.stax_extension import BroadcastTogether, Pass, ResBlock, Reshape, Resize, StaxLayer
from datasets.celeba_mask_hq import mustache_goatee_scenario
from experiment import get_baseline, get_classifiers, get_mechanisms, TrainConfig
from identifiability_tests import evaluate, print_test_results
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
tf.config.experimental.set_visible_devices([], 'GPU')

hidden_dim = 256
n_channels = hidden_dim // 4

# Classifiers
classifier_layers = \
    (ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(4, 4)),
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu)

classifier_train_config = TrainConfig(batch_size=256,
                                      optimizer=optimizers.adam(step_size=1e-3, b1=0.9),
                                      num_steps=5000,
                                      log_every=100,
                                      eval_every=100,
                                      save_every=500)

# General encoder/decoder
encoder_layers = \
    (ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)

decoder_layers = \
    (Dense(hidden_dim), LeakyRelu, Dense(8 * 8 * n_channels), LeakyRelu, Reshape((-1, 8, 8, n_channels)),
     Resize((-1, 16, 16, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
     Resize((-1, 32, 32, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
     Resize((-1, 64, 64, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
     Resize((-1, 128, 128, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
     Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))

# Conditional VAE baseline
latent_dim = 16
vae_encoder = serial(parallel(serial(*encoder_layers), Pass), FanInConcat(axis=-1),
                     Dense(hidden_dim), LeakyRelu,
                     FanOut(2), parallel(Dense(latent_dim), Dense(latent_dim)))
vae_decoder = serial(FanInConcat(axis=-1), *decoder_layers)
baseline_train_config = TrainConfig(batch_size=512,
                                    optimizer=optimizers.adam(step_size=1e-3),
                                    num_steps=10000,
                                    log_every=10,
                                    eval_every=250,
                                    save_every=250)

# Functional mechanism
critic = serial(BroadcastTogether(-1), FanInConcat(-1), *encoder_layers, Dense(hidden_dim), LeakyRelu)

mechanism = serial(parallel(serial(*encoder_layers), Pass, Pass), FanInConcat(-1),
                   Dense(hidden_dim), LeakyRelu, *decoder_layers, Tanh)

schedule = optimizers.piecewise_constant(boundaries=[5000, 10000], values=[1e-4, 1e-4 / 2, 1e-4 / 8])
mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)
mechanism_train_config = TrainConfig(batch_size=64,
                                     optimizer=mechanism_optimizer,
                                     num_steps=20000,
                                     log_every=10,
                                     eval_every=250,
                                     save_every=250)


def run_experiment(job_dir: Path,
                   data_dir: Path,
                   overwrite: bool,
                   seeds: List[int],
                   baseline: bool,
                   partial_mechanisms: bool,
                   from_joint: bool = True) -> None:
    scenario = mustache_goatee_scenario(data_dir)
    job_name = Path(f'partial_mechanisms_{partial_mechanisms}')
    scenario_name = 'mustache_goatee_scenario'
    pseudo_oracle_dir = job_dir / scenario_name / 'pseudo_oracles'
    experiment_dir = job_dir / scenario_name / job_name
    pseudo_oracles = get_classifiers(job_dir=pseudo_oracle_dir,
                                     seed=368392,
                                     scenario=scenario,
                                     classifier_layers=classifier_layers,
                                     train_config=classifier_train_config,
                                     overwrite=False)

    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    results = []
    for seed in seeds:
        seed_dir = experiment_dir / f'seed_{seed:d}'
        if baseline:
            mechanisms = get_baseline(job_dir=seed_dir,
                                      seed=seed,
                                      scenario=scenario,
                                      vae_encoder=vae_encoder,
                                      vae_decoder=vae_decoder,
                                      train_config=baseline_train_config,
                                      from_joint=from_joint,
                                      overwrite=overwrite)
        else:
            mechanisms = get_mechanisms(job_dir=seed_dir,
                                        seed=seed,
                                        scenario=scenario,
                                        partial_mechanisms=partial_mechanisms,
                                        constraint_function_power=1,
                                        classifier_layers=classifier_layers,
                                        classifier_train_config=classifier_train_config,
                                        critic=critic,
                                        mechanism=mechanism,
                                        train_config=mechanism_train_config,
                                        from_joint=from_joint,
                                        overwrite=overwrite)

        results.append(evaluate(seed_dir, mechanisms, is_invertible, marginals, pseudo_oracles, test_dataset,
                                overwrite=overwrite))

    print(job_name)
    print_test_results(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')

    args = parser.parse_args()

    run_experiment(args.job_dir,
                   args.data_dir,
                   args.overwrite,
                   args.seeds,
                   baseline=False,
                   partial_mechanisms=False)
    # for partial_mechanisms in (False, True):
