import argparse
import itertools
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import optax
import pandas as pd
import tensorflow as tf
from jax.example_libraries.stax import (Conv, Dense, FanInConcat, FanOut, Flatten, Identity, LeakyRelu, Tanh, parallel,
                                        serial)

from datasets.confounded_mnist import confoudned_mnist
from datasets.utils import Scenario
from experiment import GetModelFn, TrainConfig, get_auxiliary_models, get_counterfactual_fns
from identifiability_tests import TestResult, evaluate, format_results
from models.conditional_gan import conditional_gan
from models.conditional_vae import conditional_vae
from models.utils import AuxiliaryFn, CouterfactualFn
from staxplus import BroadcastTogether, Reshape, Resize, StaxLayer
from utils import flatten_nested_dict

tf.config.experimental.set_visible_devices([], 'GPU')

hidden_dim = 128
n_channels = 64
latent_dim = 16

# Pseudo oracle architecture and training config
aux_backbone = StaxLayer(
    *serial(
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
        Flatten,
        Dense(hidden_dim), LeakyRelu,
    )
)

aux_train_config = TrainConfig(optimizer=optax.adamw(learning_rate=1e-3, b1=0.9),
                               num_steps=4000,
                               log_every=100,
                               eval_every=200,
                               save_every=200)

# General encoder/decoder
image_encoder = serial(
    Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
    Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
    Flatten,
    Dense(hidden_dim), LeakyRelu)

image_decoder = serial(
    Dense(7 * 7 * n_channels), LeakyRelu,
    Reshape((-1, 7, 7, n_channels)),
    Resize((-1, 14, 14, n_channels), method='linear'),
    Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), LeakyRelu,
    Resize((-1, 28, 28, n_channels), method='linear'),
    Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), LeakyRelu,
    Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))


# VAE
vae_encoder = StaxLayer(
    *serial(
        parallel(image_encoder, Identity),
        FanInConcat(axis=-1),
        Dense(hidden_dim), LeakyRelu,
        FanOut(2), parallel(Dense(latent_dim), Dense(latent_dim))
    )
)
vae_decoder = StaxLayer(*serial(FanInConcat(axis=-1), image_decoder))
vae_train_config = TrainConfig(optimizer=optax.adamw(learning_rate=1e-3),
                               num_steps=20000,
                               log_every=50,
                               eval_every=500,
                               save_every=500)


# GAN
critic = StaxLayer(
    *serial(
        BroadcastTogether(), FanInConcat(-1),
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), LeakyRelu,
        Flatten, Dense(hidden_dim), LeakyRelu,
        Dense(hidden_dim), LeakyRelu
    )
)


generator = StaxLayer(
    *serial(
        parallel(image_encoder, Identity, Identity),
        FanInConcat(-1),
        Dense(hidden_dim), LeakyRelu,
        image_decoder, Tanh
    )
)
schedule = optax.piecewise_constant_schedule(init_value=1e-4, boundaries_and_scales={12000: .5, 16000: .2})
gan_optimizer = optax.adamw(learning_rate=schedule, b1=0.0, b2=.9)
gan_train_config = TrainConfig(optimizer=gan_optimizer,
                               num_steps=20000,
                               log_every=50,
                               eval_every=500,
                               save_every=500)


def get_data(data_dir: Path, data_config_path: Path) -> Tuple[str, str, Scenario, Scenario]:
    assert data_config_path.exists()
    with open(data_config_path) as f:
        config = json.load(f)

    scenario_name = str(config['scenario_name'])
    confound = bool(config['confound'])
    scale = float(config['scale'])
    outlier_prob = float(config['outlier_prob'])
    _, scenario_unconfounded = confoudned_mnist(scenario_name=scenario_name,
                                                data_dir=data_dir,
                                                batch_size=512,
                                                confound=False,
                                                scale=scale,
                                                outlier_prob=outlier_prob)
    dataset_name, scenario = confoudned_mnist(scenario_name=scenario_name,
                                              data_dir=data_dir,
                                              batch_size=512,
                                              confound=confound,
                                              scale=scale,
                                              outlier_prob=outlier_prob)
    return scenario_name, dataset_name, scenario_unconfounded, scenario


def get_vae_model(config: Dict[Any, Any]) -> Tuple[str, GetModelFn, TrainConfig]:
    bernoulli_ll = bool(config['bernoulli_ll'])
    beta = config['beta'] if 'beta' in config.keys() else 1.
    variance = float(config['normal_ll_variance']) if not bernoulli_ll else 0.
    simulated_intervention = bool(config['simulated_intervention'])
    get_model_fn = partial(conditional_vae,
                           vae_encoder=vae_encoder,
                           vae_decoder=vae_decoder,
                           bernoulli_ll=bernoulli_ll,
                           normal_ll_variance=variance,
                           beta=beta,
                           simulated_intervention=simulated_intervention)
    model_name = f'vae_beta_{beta:.1f}' + ('_bernoulli_ll' if bernoulli_ll else f'_normal_ll_variance_{variance:.2f}')
    return model_name, get_model_fn, vae_train_config


def get_gan_model(config: Dict[Any, Any]) -> Tuple[str, GetModelFn, TrainConfig]:
    use_composition_constraint = bool(config['use_composition_constraint'])
    use_reversibility_constraint = bool(config['use_reversibility_constraint'])
    get_model_fn = partial(conditional_gan,
                           critic=critic,
                           generator=generator,
                           use_composition_constraint=use_composition_constraint,
                           use_reversibility_constraint=use_reversibility_constraint,
                           simulated_intervention=bool(config['simulated_intervention']))
    model_name = 'gan' + f'_composition_constraint_{use_composition_constraint}' + \
        f'_reversibility_constraint_{use_reversibility_constraint}'
    return model_name, get_model_fn, gan_train_config


def get_model(model_config_path: Path) -> Tuple[str, bool, GetModelFn, TrainConfig]:
    assert model_config_path.exists()
    with open(model_config_path) as f:
        config = json.load(f)
    name = str(config['name'])
    simulated_intervention = bool(config['simulated_intervention'])
    if name == 'vae':
        model_name, get_model_fn, train_config = get_vae_model(config)
    elif name == 'gan':
        model_name, get_model_fn, train_config = get_gan_model(config)
    else:
        raise NotImplementedError
    model_name += f'_simulated_intervention_{simulated_intervention}'
    return model_name, simulated_intervention, get_model_fn, train_config


def get_pseudo_oracles_dict(job_dir: Path,
                            scenario: Scenario,
                            scenario_unconfounded: Scenario,
                            scenario_name: str,
                            dataset_name: str,
                            run_extra_eval: bool = False):
    base_dir = Path(job_dir) / scenario_name
    # get pseudo oracles trained from unconfounded data
    pseudo_oracles_dict = {'pseudo_oracles':  get_auxiliary_models(job_dir=base_dir / 'pseudo_oracles',
                                                                   seed=368392,
                                                                   scenario=scenario_unconfounded,
                                                                   backbone=aux_backbone,
                                                                   train_config=aux_train_config,
                                                                   overwrite=False)}
    if run_extra_eval:
        # get simple pseudo oracles trained from unconfounded data
        pseudo_oracles_dict['pseudo_oracles_simple'] \
            = get_auxiliary_models(job_dir=base_dir / 'pseudo_oracles_simple',
                                   seed=368392,
                                   scenario=scenario_unconfounded,
                                   backbone=StaxLayer(*serial(Flatten, Identity)),
                                   train_config=aux_train_config,
                                   overwrite=False)
        # get pseudo oracles trained from confounded data - more realistic scenario with and without intervention
        pseudo_oracles_dict['pseudo_oracles_confounded_with_intervention'] = \
            get_auxiliary_models(job_dir=base_dir / dataset_name / 'pseudo_oracles_confounded_with_intervention',
                                 seed=368392,
                                 scenario=scenario,
                                 backbone=aux_backbone,
                                 train_config=aux_train_config,
                                 overwrite=False,
                                 simulated_intervervention=True)
        pseudo_oracles_dict['pseudo_oracles_confounded'] = \
            get_auxiliary_models(job_dir=base_dir / dataset_name / 'pseudo_oracles_confounded',
                                 seed=368392,
                                 scenario=scenario,
                                 backbone=aux_backbone,
                                 train_config=aux_train_config,
                                 overwrite=False,
                                 simulated_intervervention=False)
    return pseudo_oracles_dict


def evaluate_mnits(job_dir: Path,
                   scenario: Scenario,
                   counterfactual_fns: Dict[str, CouterfactualFn],
                   test_data_dict: Dict[str, tf.data.Dataset],
                   pseudo_oracles_dict: Dict[str, Dict[str, AuxiliaryFn]],
                   overwrite: bool):

    result = {key0: {key1: None for key1 in pseudo_oracles_dict.keys()} for key0 in test_data_dict.keys()}
    for test_name, test_set in test_data_dict.items():
        for pseudo_oracle_name, _pseudo_oracles in pseudo_oracles_dict.items():
            result_dir = job_dir / test_name / pseudo_oracle_name
            res = evaluate(result_dir, scenario, test_set, counterfactual_fns, _pseudo_oracles, overwrite=overwrite)
            result[test_name][pseudo_oracle_name] = res
    return result


def main(job_dir: Path,
         data_dir: Path,
         data_config_path: Path,
         model_config_path: Path,
         seeds: List[int],
         run_extra_eval: bool,
         use_partial_fns: bool,
         overwrite: bool = False) -> Dict[str, Any]:

    scenario_name, dataset_name, scenario_unconfounded, scenario = get_data(data_dir, data_config_path)

    # get model
    model_name, simulated_intervention, get_model_fn, train_config = get_model(model_config_path)

    pseudo_oracles_dict = get_pseudo_oracles_dict(job_dir,
                                                  scenario,
                                                  scenario_unconfounded,
                                                  scenario_name,
                                                  dataset_name,
                                                  run_extra_eval)

    # set up testing for different setups
    if run_extra_eval:
        test_data_dict = {'confounded_test_set': scenario.test_data_confounded,
                          'unconfounded_test_set': scenario.test_data_unconfounded}
    else:
        test_data_dict = {'unconfounded_test_set': scenario.test_data_unconfounded}

    # run experiment
    experiment_dir = Path(job_dir) / scenario_name / dataset_name / model_name
    results: List[TestResult] = []
    parent_names = tuple(scenario.parent_dists.keys())
    for seed in seeds:
        seed_dir = experiment_dir / f'seed_{seed:d}'
        counterfactual_fns = get_counterfactual_fns(job_dir=seed_dir,
                                                    seed=seed,
                                                    scenario=scenario,
                                                    parent_names=parent_names,
                                                    get_model_fn=get_model_fn,
                                                    pseudo_oracles=pseudo_oracles_dict['pseudo_oracles'],
                                                    train_config=train_config,
                                                    use_partial_fns=use_partial_fns,
                                                    simulated_intervention=simulated_intervention,
                                                    overwrite=overwrite)

        results.append(evaluate_mnits(seed_dir, scenario, counterfactual_fns,
                       test_data_dict, pseudo_oracles_dict, overwrite=overwrite))

    print(experiment_dir)
    formatted_results = format_results(results, print_results=True)
    return formatted_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',
                        type=Path,
                        help='Directory where logs and models are saved.')
    parser.add_argument('--data-dir',
                        type=Path,
                        help='Directory where data files will be saved.')
    parser.add_argument('--data-config-path',
                        type=Path,
                        help='Path or directory to data config file(s). If directory all configs within will be run.')
    parser.add_argument('--model-config-path',
                        dest='model_config_path',
                        type=Path,
                        help='Path or directory to model config file(s). If directory all configs within will be run.')
    parser.add_argument('--seeds',
                        nargs='+',
                        type=int, help='List of random seeds.')
    parser.add_argument('--use_partial_fns',
                        action='store_true',
                        help='whether to train one model for each parent: partial counterfactual functions.')
    parser.add_argument('--run-extra-eval',
                        action='store_true',
                        help='whether to test on the confounded test set.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='whether to overwrite an existing run')
    args = parser.parse_args()

    data_config_path = args.data_config_path
    data_config_paths = [data_config_path] if data_config_path.is_file() else data_config_path.glob('**/*.json')
    model_config_path = args.model_config_path
    model_config_paths = [model_config_path] if model_config_path.is_file() else model_config_path.glob('**/*.json')

    all_results = {}
    for data_config_path, model_config_path in itertools.product(data_config_paths, model_config_paths):
        all_results[(data_config_path, model_config_path)] = main(job_dir=args.job_dir,
                                                                  data_dir=args.data_dir,
                                                                  data_config_path=data_config_path,
                                                                  model_config_path=model_config_path,
                                                                  seeds=args.seeds,
                                                                  use_partial_fns=args.use_partial_fns,
                                                                  run_extra_eval=args.run_extra_eval,
                                                                  overwrite=args.overwrite)

    table = pd.concat({key: pd.DataFrame.from_dict(flatten_nested_dict(value),
                      'index').T for key, value in all_results.items()}, axis=0)
    table.to_csv('/tmp/table.csv')
