import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import optax
import tensorflow as tf
import itertools
from jax.example_libraries.stax import (Conv, Dense, FanInConcat, FanOut, Flatten, Identity, LeakyRelu, Tanh, parallel,
                                        serial)

from datasets.shapes3d import shapes3d
from experiment import GetModelFn, TrainConfig, get_auxiliary_models, get_counterfactual_fns
from identifiability_tests import TestResult, evaluate, format_results
from models.conditional_gan import conditional_gan
from models.conditional_vae import conditional_vae
from staxplus import BroadcastTogether, Reshape, Resize, StaxLayer

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

aux_train_config = TrainConfig(optimizer=optax.adamw(learning_rate=5e-4, b1=0.9),
                               num_steps=4000,
                               log_every=100,
                               eval_every=500,
                               save_every=500)

# General encoder/decoder
image_encoder = serial(
    Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
    Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
    Conv(n_channels, filter_shape=(4, 4), strides=(2, 2)), LeakyRelu,
    Flatten,
    Dense(hidden_dim), LeakyRelu)

image_decoder = serial(
    Dense(8 * 8 * n_channels), LeakyRelu,
    Reshape((-1, 8, 8, n_channels)),
    Resize((-1, 16, 16, n_channels), method='linear'),
    Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), LeakyRelu,
    Resize((-1, 32, 32, n_channels), method='linear'),
    Conv(n_channels, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), LeakyRelu,
    Resize((-1, 64, 64, n_channels), method='linear'),
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
    get_model_fn = partial(conditional_gan,
                           critic=critic,
                           generator=generator,
                           simulated_intervention=bool(config['simulated_intervention']))
    return 'gan', get_model_fn, gan_train_config


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


def main(job_dir: Path,
         data_dir: Path,
         data_config_path: Path,
         model_config_path: Path,
         seeds: List[int],
         use_partial_fns: bool,
         overwrite: bool = False) -> None:

    assert data_config_path.exists()
    with open(data_config_path) as f:
        parent_names: Sequence[str] = json.load(f)['parent_names']

    scenario_name, scenario = shapes3d(data_dir, batch_size=512)
    # get pseudo oracles trained from unconfounded data
    pseudo_oracles = get_auxiliary_models(job_dir=Path(job_dir) / scenario_name / 'pseudo_oracles',
                                          seed=368392,
                                          scenario=scenario,
                                          backbone=aux_backbone,
                                          train_config=aux_train_config,
                                          overwrite=overwrite)

    model_name, simulated_intervention, get_model_fn, train_config = get_model(model_config_path)
    experiment_dir = Path(job_dir) / scenario_name / '_'.join(parent_names) / model_name
    results: List[TestResult] = []
    for seed in seeds:
        seed_dir = experiment_dir / f'seed_{seed:d}'
        counterfactual_fns = get_counterfactual_fns(job_dir=seed_dir,
                                                    seed=seed,
                                                    scenario=scenario,
                                                    parent_names=parent_names,
                                                    get_model_fn=get_model_fn,
                                                    pseudo_oracles=pseudo_oracles,
                                                    train_config=train_config,
                                                    use_partial_fns=use_partial_fns,
                                                    simulated_intervention=simulated_intervention,
                                                    overwrite=overwrite)
        test_data = scenario.test_data_unconfounded
        results.append(evaluate(seed_dir, scenario, test_data, counterfactual_fns, pseudo_oracles, overwrite=overwrite))

    print(experiment_dir)
    format_results(results)


def list_configs(_dir: Path, files: List[Path]) -> List[Path]:
    for _file in _dir.iterdir():
        if _file.is_dir():
            files += list_configs(_file, files)
        elif _file.suffix == '.json':
            files.append(_file)
    return files


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
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='whether to overwrite an existing run')
    args = parser.parse_args()

    data_config_path = args.data_config_path
    data_config_paths = [data_config_path] if data_config_path.is_file() else list_configs(data_config_path, [])
    model_config_path = args.model_config_path
    model_config_paths = [model_config_path] if model_config_path.is_file() else list_configs(model_config_path, [])

    for data_config_path, model_config_path in itertools.product(data_config_paths, model_config_paths):
        main(job_dir=args.job_dir,
             data_dir=args.data_dir,
             data_config_path=data_config_path,
             model_config_path=model_config_path,
             seeds=args.seeds,
             use_partial_fns=args.use_partial_fns,
             overwrite=args.overwrite)
