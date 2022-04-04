import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import cast, Dict, List

import tensorflow as tf
from jax.example_libraries import optimizers
from jax.example_libraries.stax import Conv, Dense, FanInConcat, FanOut, Flatten, LeakyRelu, parallel, serial, Tanh

from components.stax_extension import BroadcastTogether, Pass, PixelNorm2D, ResBlock, Reshape, Resize, StaxLayer
from datasets.confounded_mnist import digit_colour_scenario, digit_fracture_colour_scenario, Scenario
from identifiability_tests import evaluate, print_test_results
from models import classifier, ClassifierFn, functional_counterfactual, MechanismFn, vae_gan
from train import train
from utils import compile_fn, prep_classifier_data, prep_mechanism_data

tf.config.experimental.set_visible_devices([], 'GPU')

scenarios = {'digit_colour_scenario': digit_colour_scenario,
             'digit_fracture_colour_scenario': digit_fracture_colour_scenario}

hidden_dim = 64
classifier_layers = \
    (ResBlock(hidden_dim, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(hidden_dim, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(hidden_dim, filter_shape=(4, 4), strides=(2, 2)),
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)

critic = serial(BroadcastTogether(-1), FanInConcat(-1), *classifier_layers)
encoder_layers = \
    (Conv(hidden_dim, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     Conv(hidden_dim, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)

decoder_layers = \
    (Dense(7 * 7 * hidden_dim), LeakyRelu, Reshape((-1, 7, 7, hidden_dim)),
     Resize((-1, 14, 14, hidden_dim)), Conv(hidden_dim, filter_shape=(4, 4), strides=(1, 1), padding='SAME'),
     PixelNorm2D, LeakyRelu,
     Resize((-1, 28, 28, hidden_dim)), Conv(hidden_dim, filter_shape=(4, 4), strides=(1, 1), padding='SAME'),
     PixelNorm2D, LeakyRelu,
     Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))

mechanism = serial(parallel(encoder_layers,
                            serial(Dense(hidden_dim), LeakyRelu),
                            serial(Dense(hidden_dim), LeakyRelu)),
                   FanInConcat(-1), decoder_layers, Tanh)

# For the baseline C-VAE
latent_dim = 16
vae_encoder = serial(parallel(serial(*encoder_layers), Pass), FanInConcat(axis=-1),
                     Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu,
                     FanOut(2), parallel(Dense(latent_dim), serial(Dense(latent_dim))))

vae_decoder = serial(FanInConcat(axis=-1), *decoder_layers)


### VAE
# hidden_dim = 256
# n_channels = 64
#
# _enc = (Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
#         Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
#         cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)
# _dec = (Dense(hidden_dim), LeakyRelu, Dense(n_channels * 7 * 7), LeakyRelu,
#         Reshape((-1, 7, 7, n_channels)), Resize((-1, 14, 14, n_channels)),
#         Conv(n_channels, filter_shape=(5, 5), strides=(1, 1), padding='SAME'), LeakyRelu,
#         Resize((-1, 28, 28, n_channels)),
#         Conv(3, filter_shape=(5, 5), strides=(1, 1), padding='SAME'), LeakyRelu,
#         Conv(3, filter_shape=(1, 1), strides=(1, 1), padding='SAME'))
# vae_encoder = serial(parallel(serial(*_enc), Pass), FanInConcat(axis=-1),
#                      Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu,
#                      FanOut(2), parallel(Dense(latent_dim), serial(Dense(latent_dim))))
# vae_decoder = serial(FanInConcat(axis=-1), *_dec)


##

#######################################################################

# enc_init_fn, enc_apply_fn = serial(*encoder_layers)
# dec_init_fn, dec_apply_fn = serial(*decoder_layers, Tanh)
#


##

# mechanism_decoder_layers = \
#     (Dense(7 * 7 * hidden_dim), LeakyRelu, Reshape((-1, 7, 7, hidden_dim)),
#      ConvTranspose(hidden_dim, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      ConvTranspose(hidden_dim, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))

# original
# classifier_layers = \
#     (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
#      cast(StaxLayer, Flatten), Dense(128), LeakyRelu)
# mechanism_encoder_layers = \
#     (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      Reshape((-1, 7 * 7 * 128)), Dense(1024), LeakyRelu)
#
# mechanism_decoder_layers = \
#     (Dense(7 * 7 * 128), LeakyRelu, Reshape((-1, 7, 7, 128)),
#      ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))


def get_classifiers(job_dir: Path,
                    seed: int,
                    scenario: Scenario,
                    overwrite: bool) -> Dict[str, ClassifierFn]:
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    classifiers: Dict[str, ClassifierFn] = {}
    for parent_name, parent_dim in parent_dims.items():
        model = classifier(num_classes=parent_dims[parent_name], layers=classifier_layers)
        train_data, test_data = prep_classifier_data(parent_name, train_datasets, test_dataset, batch_size=1024)
        params = train(model=model,
                       job_dir=job_dir / parent_name,
                       seed=seed,
                       train_data=train_data,
                       test_data=test_data,
                       input_shape=input_shape,
                       optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                       num_steps=2000,
                       log_every=100,
                       eval_every=50,
                       save_every=50,
                       overwrite=overwrite)
        classifiers[parent_name] = compile_fn(fn=model[1], params=params)
    return classifiers


def get_baseline(job_dir: Path,
                 seed: int,
                 scenario: Scenario,
                 from_joint: bool,
                 overwrite: bool) -> Dict[str, MechanismFn]:
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    parent_names = list(parent_dims.keys())
    parent_name = 'all'
    model, get_mechanism_fn = vae_gan(parent_dims=parent_dims,
                                      marginal_dists=marginals,
                                      critic=critic,
                                      vae_encoder=vae_encoder,
                                      vae_decoder=vae_decoder,
                                      from_joint=from_joint)
    train_data, test_data = prep_mechanism_data(parent_name, parent_names, from_joint, train_datasets,
                                                test_dataset, batch_size=512)
    optimizer = optimizers.adam(step_size=1e-3)
    params = train(model=model,
                   job_dir=job_dir / f'do_{parent_name}',
                   seed=seed,
                   train_data=train_data,
                   test_data=test_data,
                   input_shape=input_shape,
                   optimizer=optimizer,
                   num_steps=10000,
                   log_every=10,
                   eval_every=250,
                   save_every=250,
                   overwrite=overwrite)
    mechanisms = {parent_name: get_mechanism_fn(params) for parent_name in parent_names}
    return mechanisms


def get_mechanisms(job_dir: Path,
                   seed: int,
                   scenario: Scenario,
                   baseline: bool,
                   partial_mechanisms: bool,
                   constraint_function_power: int,
                   from_joint: bool,
                   overwrite: bool) -> Dict[str, MechanismFn]:
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    parent_names = list(parent_dims.keys())
    classifiers = get_classifiers(job_dir / 'classifiers', seed, scenario, overwrite)
    mechanisms: Dict[str, MechanismFn] = {}
    for parent_name in (parent_names if partial_mechanisms and not baseline else ['all']):
        model, get_mechanism_fn = functional_counterfactual(do_parent_name=parent_name,
                                                            parent_dims=parent_dims,
                                                            marginal_dists=marginals,
                                                            classifiers=classifiers,
                                                            critic=critic,
                                                            mechanism=mechanism,
                                                            is_invertible=is_invertible,
                                                            constraint_function_power=constraint_function_power,
                                                            from_joint=from_joint)
        train_data, test_data = prep_mechanism_data(parent_name, parent_names, from_joint, train_datasets,
                                                    test_dataset, batch_size=512)

        schedule = optimizers.piecewise_constant(boundaries=[2000, 4000], values=[1e-4, 1e-4 / 2, 1e-4 / 8])
        optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)
        params = train(model=model,
                       job_dir=job_dir / f'do_{parent_name}',
                       seed=seed,
                       train_data=train_data,
                       test_data=test_data,
                       input_shape=input_shape,
                       optimizer=optimizer,
                       num_steps=5000,
                       log_every=10,
                       eval_every=250,
                       save_every=250,
                       overwrite=overwrite)
        mechanisms[parent_name] = get_mechanism_fn(params)
    mechanisms = {parent_name: mechanisms['all'] for parent_name in parent_names} if 'all' in mechanisms else mechanisms
    return mechanisms


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

    pseudo_oracles = get_classifiers(pseudo_oracle_dir, 100, scenario_fn(data_dir, False, False), overwrite=False)

    scenario = scenario_fn(data_dir, confound, de_confound)
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario

    results = []
    for seed in seeds:
        seed_dir = experiment_dir / f'seed_{seed:d}'
        if baseline:
            mechanisms = get_baseline(job_dir=seed_dir,
                                      seed=seed,
                                      scenario=scenario,
                                      from_joint=from_joint,
                                      overwrite=overwrite)
        else:
            mechanisms = get_mechanisms(job_dir=seed_dir,
                                        seed=seed,
                                        scenario=scenario,
                                        baseline=baseline,
                                        partial_mechanisms=partial_mechanisms,
                                        constraint_function_power=constraint_function_power,
                                        from_joint=from_joint,
                                        overwrite=overwrite)

        results.append(evaluate(seed_dir, mechanisms, is_invertible, marginals, pseudo_oracles, test_dataset,
                                overwrite=overwrite))
    print(job_name)
    print_test_results(results)


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
    configs = [Config(baseline, partial_mechanisms, constraint_function_power, confound, de_confound)
               for baseline, partial_mechanisms, constraint_function_power, (confound, de_confound)
               in product((False, True), (False, True), (1, 3), ((True, True), (False, False)))]

    configs = [Config(True, False, 1, False, False)]

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
