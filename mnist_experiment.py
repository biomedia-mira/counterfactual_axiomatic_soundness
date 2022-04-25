import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import optax
import tensorflow as tf
from jax.example_libraries.stax import (Conv, Dense, FanInConcat, FanOut, Flatten, Identity, LeakyRelu, Tanh, parallel,
                                        serial)

from datasets.confounded_mnist import confounded_mnist
from experiment import TrainConfig, get_baseline, get_discriminative_models, get_mechanisms
from identifiability_tests import evaluate, print_test_results
from staxplus import Reshape, Resize, StaxLayer

tf.config.experimental.set_visible_devices([], 'GPU')

hidden_dim = 256
n_channels = hidden_dim // 4

# Discriminative model layers
discriminative_backbone \
    = StaxLayer(*serial(
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Flatten,
        Dense(hidden_dim), LeakyRelu,
        Dense(hidden_dim), LeakyRelu))

discriminative_train_config = TrainConfig(batch_size=1024,
                                          optimizer=optax.adam(learning_rate=5e-4, b1=0.9),
                                          num_steps=2000,
                                          log_every=100,
                                          eval_every=50,
                                          save_every=50)

# General encoder/decoder
encoder_layers = (Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
                  Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
                  Flatten, Dense(hidden_dim), LeakyRelu)

decoder_layers = \
    (Dense(hidden_dim), LeakyRelu, Dense(7 * 7 * n_channels), LeakyRelu, Reshape((-1, 7, 7, n_channels)),
     Resize((-1, 14, 14, n_channels)), Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), LeakyRelu,
     Resize((-1, 28, 28, n_channels)), Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), LeakyRelu,
     Conv(1, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))

# Conditional VAE baseline
latent_dim = 16
vae_encoder = StaxLayer(*serial(parallel(serial(*encoder_layers), Identity), FanInConcat(axis=-1),
                                Dense(hidden_dim), LeakyRelu,
                                FanOut(2), parallel(Dense(latent_dim), Dense(latent_dim))))
vae_decoder = StaxLayer(*serial(FanInConcat(axis=-1), *decoder_layers))
baseline_train_config = TrainConfig(batch_size=512,
                                    optimizer=optax.adam(learning_rate=1e-3),
                                    num_steps=10000,
                                    log_every=10,
                                    eval_every=250,
                                    save_every=250)

critic = StaxLayer(*serial(
    parallel(
        serial(
            Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
            Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
            Flatten, Dense(hidden_dim), LeakyRelu),
        serial(Dense(hidden_dim), LeakyRelu)
    ),
    FanInConcat(-1),
    Dense(hidden_dim), LeakyRelu,
    Dense(hidden_dim), LeakyRelu))

mechanism = StaxLayer(*serial(
    parallel(serial(*encoder_layers), Identity, Identity),
    FanInConcat(-1),
    Dense(hidden_dim), LeakyRelu,
    *decoder_layers, Tanh))

mechanism_optimizer = optax.chain(optax.adam(learning_rate=1e-4, b1=0.0, b2=.9),
                                  optax.adaptive_grad_clip(clipping=0.01))
mechanism_train_config = TrainConfig(batch_size=512,
                                     optimizer=mechanism_optimizer,
                                     num_steps=10000,
                                     log_every=10,
                                     eval_every=250,
                                     save_every=250)


def run_experiment(job_dir: Path,
                   data_dir: Path,
                   scenario_name: str,
                   overwrite: bool,
                   seeds: List[int],
                   baseline: bool,
                   partial_mechanisms: bool,
                   constraint_function_power: int,
                   confound: bool,
                   from_joint: bool = True) -> None:
    prefix = 'baseline' if baseline else f'partial_mechanisms_{partial_mechanisms}_M_{constraint_function_power:d}'
    suffix = f'_confounded_from_joint_{from_joint}' if confound else 'not_confounded'
    job_name = Path(prefix + suffix)
    pseudo_oracle_dir = job_dir / scenario_name / 'pseudo_oracles'
    experiment_dir = job_dir / scenario_name / job_name

    scenario_unconfounded = confounded_mnist(data_dir, scenario_name, confound=False)
    scenario = confounded_mnist(data_dir, scenario_name, confound=confound)
    _, test_data, parent_dists, _ = scenario
    # get pseudo oracles
    pseudo_oracles = get_discriminative_models(job_dir=pseudo_oracle_dir,
                                               seed=368392,
                                               scenario=scenario_unconfounded,
                                               backbone=discriminative_backbone,
                                               train_config=discriminative_train_config,
                                               overwrite=False)

    # ood_test_sets= {'kmnist': get_coloured_kmnist(data_dir, True)} if scenario_name == 'digit_colour_scenario' else {}
    ood_test_sets = {}
    ood_results = {key: [] for key in ood_test_sets.keys()}
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
                                        constraint_function_power=constraint_function_power,
                                        discriminative_backbone=discriminative_backbone,
                                        classifier_train_config=discriminative_train_config,
                                        critic=critic,
                                        mechanism=mechanism,
                                        train_config=mechanism_train_config,
                                        from_joint=from_joint,
                                        overwrite=overwrite)

        results.append(evaluate(seed_dir, parent_dists, mechanisms, pseudo_oracles, test_data, overwrite=overwrite))
        for key, ood_test_set in ood_test_sets.items():
            ood_results[key].append(evaluate(seed_dir / 'ood', parent_dists, mechanisms, pseudo_oracles,
                                             ood_test_set, overwrite=overwrite))

    print(job_name)
    print_test_results(results)
    for key, ood_test_result in ood_results.items():
        print(key)
        print_test_results(ood_test_result)


@dataclass
class Config:
    baseline: bool
    partial_mechanisms: bool
    constraint_function_power: int
    confound: bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--scenario-name', dest='scenario_name', type=str, help='Name of scenario to run.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')

    args = parser.parse_args()

    configs = [
        Config(baseline=False, partial_mechanisms=False, constraint_function_power=1, confound=True),
    ]

    for config in configs:
        run_experiment(args.job_dir,
                       args.data_dir,
                       args.scenario_name,
                       args.overwrite,
                       args.seeds,
                       baseline=config.baseline,
                       partial_mechanisms=config.partial_mechanisms,
                       constraint_function_power=config.constraint_function_power,
                       confound=config.confound)
