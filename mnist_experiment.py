import argparse
from pathlib import Path

import jax.random as random
import optax
import tensorflow as tf
from jax.example_libraries.stax import (Conv, Dense, FanInConcat, FanOut, Flatten, Identity, LeakyRelu, Tanh, parallel,
                                        serial)

from datasets.confounded_mnist import confoudned_mnist
from experiment import TrainConfig, get_baseline, get_auxiliary_models, get_mechanisms
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
     Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))

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


def main(job_dir: Path,
         data_dir: Path,
         scenario_name: str,
         confound: bool = True,
         scale: float = .5,
         outlier_prob: float = 0.01,
         baseline: bool = True,
         partial_mechanisms: bool = False,
         constraint_function_power: int = 1,
         from_joint: bool = True,
         overwrite: bool = False,
         seed: int = 1,
         num_seeds: int = 1) -> None:

    _, scenario_unconfounded \
        = confoudned_mnist(scenario_name, data_dir, confound=False, scale=scale, outlier_prob=outlier_prob)
    dataset_name, scenario \
        = confoudned_mnist(scenario_name, data_dir, confound=confound, scale=scale, outlier_prob=outlier_prob)

    job_name = 'cvae' if baseline else f'func_mech_partial_{partial_mechanisms}' \
        f'_M_{constraint_function_power:d}_from_joint_{from_joint}'
    job_name += f'_{dataset_name}'

    pseudo_oracle_dir = Path(job_dir) / scenario_name / 'pseudo_oracles'
    experiment_dir = Path(job_dir) / scenario_name / job_name

    # get pseudo oracles
    pseudo_oracles = get_auxiliary_models(job_dir=pseudo_oracle_dir,
                                          seed=368392,
                                          scenario=scenario_unconfounded,
                                          backbone=discriminative_backbone,
                                          train_config=discriminative_train_config,
                                          overwrite=False)

    results = []
    seeds = [int(k[1]) for k in random.split(random.PRNGKey(seed), num_seeds)]
    for _seed in seeds:
        seed_dir = experiment_dir / f'seed_{_seed:d}'
        if baseline:
            mechanisms = get_baseline(job_dir=seed_dir,
                                      seed=_seed,
                                      scenario=scenario,
                                      vae_encoder=vae_encoder,
                                      vae_decoder=vae_decoder,
                                      train_config=baseline_train_config,
                                      from_joint=from_joint,
                                      overwrite=overwrite)
        else:
            mechanisms = get_mechanisms(job_dir=seed_dir,
                                        seed=_seed,
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

        results.append(evaluate(seed_dir, scenario, mechanisms, pseudo_oracles, overwrite=overwrite))

    print(job_name)
    print_test_results(results)


if __name__ == '__main__':

    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() == 'true':
            return True
        elif v.lower() == 'false':
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--scenario-name', dest='scenario_name', type=str, help='Name of scenario to run.')
    parser.add_argument("--confound", type=str2bool, nargs='?', const=True, default=True, help='Confound the data.')
    parser.add_argument('--scale', dest='scale', type=float, default=.5, help='Scale used for confounding function.')
    parser.add_argument('--outlier-prob', dest='outlier_prob', type=float, default=.01, help='Probablity of outliers.')
    parser.add_argument('--baseline', action='store_true', help='Use cvae baseline instead of model.')
    parser.add_argument('--partial_mechanisms', action='store_true', help='Use partial mechanisms.')
    parser.add_argument('--constraint_function_power', dest='constraint_function_power', type=int, default=1, help='')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seed', dest='seed', type=int, help='Random seed.')
    parser.add_argument('--num-seeds', dest='num_seeds', type=int, help='Number of splits for the random seed.')

    args = parser.parse_args()

    main(job_dir=args.job_dir,
         data_dir=args.data_dir,
         scenario_name=args.scenario_name,
         confound=args.confound,
         scale=args.scale,
         baseline=args.baseline,
         partial_mechanisms=args.partial_mechanisms,
         constraint_function_power=args.constraint_function_power,
         overwrite=args.overwrite,
         seed=args.seed,
         num_seeds=args.num_seeds)
