import argparse
from pathlib import Path
from typing import Any, cast, Dict, List, Tuple

import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import tensorflow as tf
from flaxmodels.stylegan2.discriminator import Discriminator as StyleGanDiscriminator
from flaxmodels.stylegan2.generator import Generator as StyleGanGenerator
from jax.example_libraries.stax import Dense, Flatten, LeakyRelu, serial

from core import Array, KeyArray, Params, Shape, ShapeTree
from core.staxplus import ResBlock, StaxLayer
from datasets.celeba_mask_hq import mustache_goatee_scenario
from experiment import get_discriminative_models, get_mechanisms, TrainConfig
from identifiability_tests import evaluate, print_test_results

tf.config.experimental.set_visible_devices([], 'GPU')

# Classifiers
hidden_dim = 256
n_channels = hidden_dim // 4
classifier_layers = \
    (ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(4, 4)),
     cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu)

classifier_train_config = TrainConfig(batch_size=256,
                                      optimizer=optax.adam(learning_rate=1e-3, b1=0.9),
                                      num_steps=5000,
                                      log_every=100,
                                      eval_every=100,
                                      save_every=500)

mechanism_train_config = TrainConfig(batch_size=16,
                                     optimizer=optax.adam(learning_rate=0.0025, b1=0., b2=.99),
                                     num_steps=40000,
                                     log_every=10,
                                     eval_every=5000,
                                     save_every=1000)


def critic(resolution: int, parent_dims: Dict[str, int]):
    c_dim = sum(parent_dims.values())
    discriminator = StyleGanDiscriminator(resolution=resolution, c_dim=c_dim, mbstd_group_size=8)

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Tuple[Shape, Params]:
        dummy_x, dummy_c = jnp.zeros((1, *input_shape[0][1:])), jnp.zeros((1, *input_shape[1][1:]))
        output, params = discriminator.init_with_output(rng, x=dummy_x, c=dummy_c)
        return output.shape, params

    def apply_fn(params: Params, inputs: Any, **kwargs) -> Array:
        return discriminator.apply(params, x=inputs[0], c=inputs[1])

    return init_fn, apply_fn


# ignores moving_stats and noise_const because they are not being used, if they were this code would be wrong
def mechanism(resolution: int,
              parent_dims: Dict[str, int],
              z_dim: int = 1024,
              w_dim: int = 512):
    c_dim = 2 * sum(parent_dims.values())
    img_enc_init_fn, img_enc_apply_fn = serial(ResBlock(16, filter_shape=(4, 4), strides=(2, 2)),
                                               ResBlock(32, filter_shape=(4, 4), strides=(2, 2)),
                                               ResBlock(64, filter_shape=(4, 4), strides=(2, 2)),
                                               ResBlock(128, filter_shape=(4, 4), strides=(2, 2)),
                                               cast(StaxLayer, Flatten), Dense(z_dim), LeakyRelu)

    generator = StyleGanGenerator(resolution=resolution, z_dim=z_dim, c_dim=c_dim,
                                  w_dim=w_dim, num_ws=int(np.log2(resolution)) * 2 - 3, num_mapping_layers=2)

    def init_fn(rng: KeyArray, input_shape: ShapeTree) -> Tuple[Shape, Params]:
        assert input_shape[1] == input_shape[2] and len(input_shape[1]) == 2 and input_shape[1][-1] == c_dim // 2
        k1, k2 = random.split(rng, 2)
        z_shape, enc_params = img_enc_init_fn(k1, input_shape[0])

        dummy_z = jnp.zeros((1, *z_shape[1:]))
        dummy_c = jnp.zeros((1, c_dim))
        output, gen_params = generator.init_with_output(rng, z=dummy_z, c=dummy_c)

        return output.shape, (enc_params, gen_params)

    def apply_fn(params: Params, inputs: Any) -> Array:
        image, parents, do_parents = inputs
        z = img_enc_apply_fn(params[0], image)
        c = jnp.concatenate((parents, do_parents), axis=-1)
        #TODO: add rng to generator
        fake_image, _ = generator.apply(params[1], z=z, c=c, mutable='moving_stats')
        return fake_image

    return init_fn, apply_fn


def run_experiment(job_dir: Path,
                   data_dir: Path,
                   overwrite: bool,
                   seeds: List[int],
                   partial_mechanisms: bool,
                   from_joint: bool = True) -> None:
    scenario = mustache_goatee_scenario(data_dir)
    job_name = Path(f'partial_mechanisms_{partial_mechanisms}')
    scenario_name = 'mustache_goatee_scenario'
    pseudo_oracle_dir = job_dir / scenario_name / 'pseudo_oracles'
    experiment_dir = job_dir / scenario_name / job_name
    pseudo_oracles = get_discriminative_models(job_dir=pseudo_oracle_dir,
                                               seed=368392,
                                               scenario=scenario,
                                               layers=classifier_layers,
                                               train_config=classifier_train_config,
                                               overwrite=False)

    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    resolution = input_shape[1]
    results = []
    for seed in seeds:
        seed_dir = experiment_dir / f'seed_{seed:d}'

        mechanisms = get_mechanisms(job_dir=seed_dir,
                                    seed=seed,
                                    scenario=scenario,
                                    partial_mechanisms=partial_mechanisms,
                                    constraint_function_power=1,
                                    classifier_layers=classifier_layers,
                                    classifier_train_config=classifier_train_config,
                                    critic=critic(resolution, parent_dims),
                                    mechanism=mechanism(resolution, parent_dims),
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
    parser.add_argument('--scenario-name', dest='scenario_name', type=str, help='Name of scenario to run.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')
    args = parser.parse_args()

    run_experiment(args.job_dir,
                   args.data_dir,
                   args.overwrite,
                   args.seeds,
                   partial_mechanisms=False)
