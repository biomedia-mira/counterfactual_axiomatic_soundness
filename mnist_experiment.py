import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast, List

import tensorflow as tf
from jax.example_libraries import optimizers
from jax.example_libraries.stax import Conv, Dense, FanInConcat, FanOut, Flatten, LeakyRelu, parallel, serial, Tanh

from components.stax_extension import BroadcastTogether, Pass, PixelNorm2D, ResBlock, Reshape, Resize, StaxLayer
from datasets.confounded_mnist import digit_colour_scenario, digit_fracture_colour_scenario, \
    digit_thickness_colour_scenario
from datasets.mnist_ood import get_coloured_kmnist
from experiment import get_baseline, get_classifiers, get_mechanisms, TrainConfig
from identifiability_tests import evaluate, print_test_results

tf.config.experimental.set_visible_devices([], 'GPU')

scenarios = {'digit_colour_scenario': digit_colour_scenario,
             'digit_fracture_colour_scenario': digit_fracture_colour_scenario,
             'digit_thickness_colour_scenario': digit_thickness_colour_scenario}

hidden_dim = 256
n_channels = hidden_dim // 4

# Classifiers
classifier_layers = \
    (Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
     Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
     Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu)

classifier_train_config = TrainConfig(batch_size=1024,
                                      optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                                      num_steps=2000,
                                      log_every=100,
                                      eval_every=50,
                                      save_every=50)

# General encoder/decoder
encoder_layers = \
    (Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)

decoder_layers = \
    (Dense(hidden_dim), LeakyRelu, Dense(7 * 7 * n_channels), LeakyRelu, Reshape((-1, 7, 7, n_channels)),
     Resize((-1, 14, 14, n_channels)), Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'),
     PixelNorm2D, LeakyRelu,
     Resize((-1, 28, 28, n_channels)), Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'),
     PixelNorm2D, LeakyRelu,
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
critic = serial(BroadcastTogether(-1), FanInConcat(-1),
                ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
                ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
                ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
                cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)
mechanism = serial(parallel(serial(*encoder_layers), Pass, Pass), FanInConcat(-1),
                   Dense(hidden_dim), LeakyRelu, *decoder_layers, Tanh)

schedule = optimizers.piecewise_constant(boundaries=[3000, 6000], values=[1e-4, 1e-4 / 2, 1e-4 / 8])
mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)
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
                   confound: bool = True,
                   de_confound: bool = True,
                   from_joint: bool = True) -> None:
    scenario_fn = scenarios[scenario_name]
    prefix = 'baseline' if baseline else f'partial_mechanisms_{partial_mechanisms}_M_{constraint_function_power:d}'
    suffix = f'_confound_{confound}_de_confounded_{de_confound}_from_joint_{from_joint}'
    job_name = Path(prefix + suffix)
    pseudo_oracle_dir = job_dir / scenario_name / 'pseudo_oracles'
    experiment_dir = job_dir / scenario_name / job_name

    pseudo_oracles = get_classifiers(job_dir=pseudo_oracle_dir,
                                     seed=368392,
                                     scenario=scenario_fn(data_dir, False, False),
                                     classifier_layers=classifier_layers,
                                     train_config=classifier_train_config,
                                     overwrite=False)

    scenario = scenario_fn(data_dir, confound, de_confound)
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario

    ood_test_sets = {'kmnist': get_coloured_kmnist(data_dir, True)} if scenario_name == 'digit_colour_scenario' else {}
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
                                        classifier_layers=classifier_layers,
                                        classifier_train_config=classifier_train_config,
                                        critic=critic,
                                        mechanism=mechanism,
                                        train_config=mechanism_train_config,
                                        from_joint=from_joint,
                                        overwrite=overwrite)

        results.append(evaluate(seed_dir, mechanisms, is_invertible, marginals, pseudo_oracles, test_dataset,
                                overwrite=overwrite))
        for key, ood_test_set in ood_test_sets.items():
            ood_results[key].append(evaluate(seed_dir / 'ood', mechanisms, is_invertible, marginals, pseudo_oracles,
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
    de_confound: bool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--scenario-name', dest='scenario_name', type=str, help='Name of scenario to run.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')

    args = parser.parse_args()
    # configs = [
    #     Config(baseline=True, partial_mechanisms=False, constraint_function_power=1, confound=True, de_confound=True),
    #     Config(baseline=True, partial_mechanisms=False, constraint_function_power=1, confound=False, de_confound=False),
    #     Config(baseline=False, partial_mechanisms=True, constraint_function_power=1, confound=True, de_confound=True),
    #     Config(baseline=False, partial_mechanisms=True, constraint_function_power=1, confound=False, de_confound=False),
    #     Config(baseline=False, partial_mechanisms=False, constraint_function_power=1, confound=True, de_confound=True),
    #     Config(baseline=False, partial_mechanisms=False, constraint_function_power=1, confound=False, de_confound=False)
    # ]
    #
    configs = [
        Config(baseline=True, partial_mechanisms=False, constraint_function_power=1, confound=True, de_confound=True),
        Config(baseline=False, partial_mechanisms=True, constraint_function_power=1, confound=True, de_confound=True),
        Config(baseline=False, partial_mechanisms=False, constraint_function_power=1, confound=True, de_confound=True),
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
                       confound=config.confound,
                       de_confound=config.de_confound)
