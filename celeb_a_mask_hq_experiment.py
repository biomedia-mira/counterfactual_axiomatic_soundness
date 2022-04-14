import argparse
from pathlib import Path
from typing import Any, Dict, Tuple
from typing import cast, List

import jax
import jax.numpy as jnp
import jax.random as random
import tensorflow as tf
from flaxmodels.stylegan2.discriminator import Discriminator as style_gan_discriminator
from flaxmodels.stylegan2.generator import Generator as style_gan_generator
from jax import jit, value_and_grad
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import adam
from jax.example_libraries.optimizers import Optimizer, OptimizerState, ParamsFn
from jax.example_libraries.stax import Conv, Dense, FanInConcat, FanOut, Flatten, LeakyRelu, parallel, serial, Tanh
from jax.lax import stop_gradient

from components import Array, KeyArray, Model, Params, Shape, StaxLayer, UpdateFn
from components.stax_extension import BroadcastTogether, Pass, ResBlock, Reshape, Resize
from datasets.celeba_mask_hq import mustache_goatee_scenario
from experiment import get_baseline, get_classifiers, get_mechanisms
from experiment import prep_mechanism_data, TrainConfig
from identifiability_tests import evaluate, print_test_results
from models.utils import MechanismFn

tf.config.experimental.set_visible_devices([], 'GPU')


def style_gan_model(z_dim: int = 512) -> Model:
    generator = style_gan_generator(resolution=128,
                                    z_dim=z_dim,
                                    c_dim=0,
                                    w_dim=512)

    discriminator = style_gan_discriminator(resolution=128, c_dim=0, mbstd_group_size=8)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Any]:
        k1, k2 = random.split(rng, 2)
        generator_vars = generator.init(k1, jnp.zeros((1, z_dim)))
        discriminator_vars = discriminator.init(k2, jnp.zeros((1, *input_shape[1:])))
        return (), (generator_vars, discriminator_vars)

    def apply_fn(vars: Any, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        generator_vars, discriminator_vars = vars
        k1, k2 = random.split(rng, 2)
        image, parents = inputs[frozenset()]
        z = random.normal(k1, shape=(image.shape[0], z_dim))
        fake_image = generator.apply(generator_vars, z, rng=k2)
        return z, {'image': image, 'fake_image': fake_image}

    def step_generator(vars: Any, inputs: Any, rng: KeyArray):
        def loss_fn(gen_params):
            image, parents = inputs[frozenset()]
            generator_vars, discriminator_vars = vars
            z = random.normal(rng, (image.shape[0], z_dim))
            _generator_vars = {**generator_vars, 'params': gen_params}
            fake_image, moving_stats = generator.apply(_generator_vars, z, rng=rng, mutable='moving_stats')
            fake_logits = discriminator.apply(stop_gradient(discriminator_vars), fake_image)
            loss = jnp.mean(jax.nn.softplus(-fake_logits))
            return loss, moving_stats, {'fake_image': fake_image, 'gen_loss': loss[jnp.newaxis]}

        (loss, moving_stats, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(vars[0].params)
        return (loss, moving_stats, output), grads

    def step_discriminator(params: Params, inputs: Any, rng: KeyArray):
        image, parents = inputs[frozenset()]
        generator_params, discriminator_params = params
        z = random.normal(rng, (image.shape[0], z_dim))
        fake_image = generator_apply_fn(generator_params, z, rng=rng)
        fake_logits = discriminator_apply_fn(stop_gradient(discriminator_params), fake_image)
        real_logits = discriminator_apply_fn(discriminator_params, inputs)
        loss_fake = jax.nn.softplus(fake_logits)
        loss_real = jax.nn.softplus(-real_logits)
        loss = jnp.mean(loss_fake + loss_real)
        return loss, {'loss_real': loss_real, 'loss_fake': loss_fake, 'loss': loss[jnp.newaxis]}

    def init_optimizer_fn(vars: Any, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizer
        opt_state = opt_init((vars[0].params, vars[1].params))

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            k1, k2 = random.split(rng, 2)
            (disc_loss, disc_outputs), disc_grads \
                = value_and_grad(step_discriminator, has_aux=True)(get_params(opt_state), inputs=inputs, rng=k1)
            opt_state = opt_update(i, disc_grads, opt_state)
            (gen_loss, gen_outputs, gen_moving_stats), gen_grads \
                = value_and_grad(step_generator, has_aux=True)(get_params(opt_state), inputs=inputs, rng=k2)
            opt_state = opt_update(i, gen_grads, opt_state)

            return opt_state, disc_loss + gen_loss, {**disc_outputs, **gen_outputs}

        return opt_state, update, get_params

    return init_fn, apply_fn, init_optimizer_fn


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

# General encoder / decoder
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
    scenario = mustache_goatee_scenario(args.data_dir)
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    parent_names = list(parent_dims.keys())

    mechanisms: Dict[str, MechanismFn] = {}
    model = style_gan_model(z_dim=512)
    train_config = TrainConfig(8, adam(step_size=0.0025, b1=0., b2=.99), 20000, 10, 1000, 1000)
    train_data, test_data = prep_mechanism_data('all', parent_names, True, train_datasets,
                                                test_dataset, batch_size=train_config.batch_size)

    # params = train(model=model,
    #                job_dir=Path('/tmp/test_style_gan'),
    #                seed=8653453,
    #                train_data=train_data,
    #                test_data=test_data,
    #                input_shape=input_shape,
    #                optimizer=train_config.optimizer,
    #                num_steps=train_config.num_steps,
    #                log_every=train_config.log_every,
    #                eval_every=train_config.eval_every,
    #                save_every=train_config.save_every,
    #                overwrite=True)

    # run_experiment(args.job_dir,
    #                args.data_dir,
    #                args.overwrite,
    #                args.seeds,
    #                baseline=False,
    #                partial_mechanisms=False)
    # for partial_mechanisms in (False, True):
