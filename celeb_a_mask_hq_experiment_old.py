import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import tensorflow as tf
from flaxmodels.stylegan2.discriminator import Discriminator as StyleGanDiscriminator
from flaxmodels.stylegan2.generator import Generator as StyleGanGenerator
from jax.lax import stop_gradient

from staxplus import Array, GradientTransformation, KeyArray, Model, OptState, Params, Shape
from staxplus.train import train
from datasets.celeba_mask_hq import mustache_goatee_scenario
from experiment import prep_mechanism_data, TrainConfig
from models.utils import MechanismFn

tf.config.experimental.set_visible_devices([], 'GPU')


def style_gan_discriminator(resolution: int, num_channels: int = 3, c_dim: int = 0, mbstd_group_size: int = None):
    discriminator = StyleGanDiscriminator(resolution=resolution, num_channels=num_channels, c_dim=c_dim,
                                          mbstd_group_size=mbstd_group_size)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        output, params = discriminator.init_with_output(rng, jnp.zeros((1, *input_shape[1:])))
        return output.shape, params

    def apply_fn(params: Params, inputs: Any) -> Array:
        return discriminator.apply(params, inputs)

    return init_fn, apply_fn


# ignores moving_stats and noise_const because they are not being used, if they were this code would be wrong
def style_gan_generator(resolution: int,
                        num_channels: int = 3,
                        z_dim: int = 512,
                        c_dim: int = 0,
                        w_dim: int = 512):
    generator = StyleGanGenerator(resolution=resolution, num_channels=num_channels, z_dim=z_dim, c_dim=c_dim,
                                  w_dim=w_dim, num_ws=int(np.log2(resolution)) * 2 - 3)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        assert input_shape[1] == z_dim and len(input_shape) == 2
        output, params = generator.init_with_output(rng, jnp.zeros((1, z_dim)))
        return output.shape, params

    def apply_fn(params: Params, z, rng: KeyArray) -> Array:
        fake_image, _ = generator.apply(params, z, rng=rng, mutable='moving_stats')
        return fake_image

    return init_fn, apply_fn


def style_gan_model(resolution: int, z_dim: int = 512, c_dim=0, w_dim=512) -> Model:
    generator_init_fn, generator_apply_fn \
        = style_gan_generator(resolution=resolution, z_dim=z_dim, c_dim=c_dim, w_dim=w_dim)
    discriminator_init_fn, discriminator_apply_fn \
        = style_gan_discriminator(resolution=resolution, c_dim=c_dim, mbstd_group_size=8)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = random.split(rng, 2)
        output_shape, generator_params = generator_init_fn(k1, (-1, z_dim))
        _, discriminator_params = discriminator_init_fn(k2, input_shape)
        return output_shape, (generator_params, discriminator_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        image, parents = inputs[frozenset()]
        k1, k2 = random.split(rng, 2)
        z = random.normal(k1, shape=(image.shape[0], z_dim))
        fake_image = generator_apply_fn(params[0], z, k2)
        return fake_image, {'image': image, 'fake_image': fake_image}

    def step_generator(params: Params, inputs: Any, rng: KeyArray):
        image, parents = inputs[frozenset()]
        k1, k2 = random.split(rng, 2)
        z = random.normal(k1, shape=(image.shape[0], z_dim))
        fake_image = generator_apply_fn(params[0], z, k2)
        fake_logits = discriminator_apply_fn(stop_gradient(params[1]), fake_image)
        loss = jnp.mean(jax.nn.softplus(-fake_logits))
        # fn = lambda *args: jnp.sum(generator_apply_fn(*args) * random.normal(k1, shape=image.shape) / resolution ** 2.)
        # pl_grads = jax.grad(fn)(params[0], z, k2)
        return loss, {'image': image, 'fake_image': fake_image, 'gen_loss': loss[jnp.newaxis]}

    def step_discriminator(params: Params, inputs: Any, rng: KeyArray):
        image, parents = inputs[frozenset()]
        k1, k2 = random.split(rng, 2)
        z = random.normal(k1, (image.shape[0], z_dim))
        fake_image = generator_apply_fn(stop_gradient(params[0]), z, k2)
        fake_logits = discriminator_apply_fn(params[1], fake_image)
        real_logits = discriminator_apply_fn(params[1], image)
        loss_fake = jax.nn.softplus(fake_logits)
        loss_real = jax.nn.softplus(-real_logits)
        loss = jnp.mean(loss_fake + loss_real)
        return loss, {'disc_loss': loss[jnp.newaxis]}

    def update(params: Params, optimizer: GradientTransformation, opt_state: OptState, inputs: Any, rng: KeyArray) \
            -> Tuple[Params, OptState, Array, Any]:
        k1, k2 = random.split(rng, 2)
        # Step discriminator
        (disc_loss, disc_output), grads = jax.value_and_grad(step_discriminator, has_aux=True)(params, inputs, k1)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # Step generator
        (gen_loss, gen_output), grads = jax.value_and_grad(step_generator, has_aux=True)(params, inputs, k2)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, disc_loss + gen_loss, {**disc_output, **gen_output}

    return init_fn, apply_fn, update


# hidden_dim = 256
# n_channels = hidden_dim // 4
#
# # Classifiers
# classifier_layers = \
#     (ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(hidden_dim // 2, filter_shape=(4, 4), strides=(4, 4)),
#      cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu, Dense(hidden_dim), LeakyRelu)
#
# classifier_train_config = TrainConfig(batch_size=256,
#                                       optimizer=optimizers.adam(step_size=1e-3, b1=0.9),
#                                       num_steps=5000,
#                                       log_every=100,
#                                       eval_every=100,
#                                       save_every=500)
#
# # General encoder / decoder
# encoder_layers = \
#     (ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
#      ResBlock(n_channels, filter_shape=(4, 4), strides=(2, 2)),
#      cast(StaxLayer, Flatten), Dense(hidden_dim), LeakyRelu)
#
# decoder_layers = \
#     (Dense(hidden_dim), LeakyRelu, Dense(8 * 8 * n_channels), LeakyRelu, Reshape((-1, 8, 8, n_channels)),
#      Resize((-1, 16, 16, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
#      Resize((-1, 32, 32, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
#      Resize((-1, 64, 64, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
#      Resize((-1, 128, 128, n_channels)), ResBlock(n_channels, filter_shape=(4, 4), strides=(1, 1)),
#      Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))
#
# # Conditional VAE baseline
# latent_dim = 16
# vae_encoder = serial(parallel(serial(*encoder_layers), Pass), FanInConcat(axis=-1),
#                      Dense(hidden_dim), LeakyRelu,
#                      FanOut(2), parallel(Dense(latent_dim), Dense(latent_dim)))
# vae_decoder = serial(FanInConcat(axis=-1), *decoder_layers)
# baseline_train_config = TrainConfig(batch_size=512,
#                                     optimizer=optimizers.adam(step_size=1e-3),
#                                     num_steps=10000,
#                                     log_every=10,
#                                     eval_every=250,
#                                     save_every=250)
#
# # Functional mechanism
# critic = serial(BroadcastTogether(-1), FanInConcat(-1), *encoder_layers, Dense(hidden_dim), LeakyRelu)
#
# mechanism = serial(parallel(serial(*encoder_layers), Pass, Pass), FanInConcat(-1),
#                    Dense(hidden_dim), LeakyRelu, *decoder_layers, Tanh)
#
# schedule = optimizers.piecewise_constant(boundaries=[5000, 10000], values=[1e-4, 1e-4 / 2, 1e-4 / 8])
# mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)
# mechanism_train_config = TrainConfig(batch_size=64,
#                                      optimizer=mechanism_optimizer,
#                                      num_steps=20000,
#                                      log_every=10,
#                                      eval_every=250,
#                                      save_every=250)


# def run_experiment(job_dir: Path,
#                    data_dir: Path,
#                    overwrite: bool,
#                    seeds: List[int],
#                    baseline: bool,
#                    partial_mechanisms: bool,
#                    from_joint: bool = True) -> None:
#     scenario = mustache_goatee_scenario(data_dir)
#     job_name = Path(f'partial_mechanisms_{partial_mechanisms}')
#     scenario_name = 'mustache_goatee_scenario'
#     pseudo_oracle_dir = job_dir / scenario_name / 'pseudo_oracles'
#     experiment_dir = job_dir / scenario_name / job_name
#     pseudo_oracles = get_classifiers(job_dir=pseudo_oracle_dir,
#                                      seed=368392,
#                                      scenario=scenario,
#                                      classifier_layers=classifier_layers,
#                                      train_config=classifier_train_config,
#                                      overwrite=False)
#
#     train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
#     results = []
#     for seed in seeds:
#         seed_dir = experiment_dir / f'seed_{seed:d}'
#         if baseline:
#             mechanisms = get_baseline(job_dir=seed_dir,
#                                       seed=seed,
#                                       scenario=scenario,
#                                       vae_encoder=vae_encoder,
#                                       vae_decoder=vae_decoder,
#                                       train_config=baseline_train_config,
#                                       from_joint=from_joint,
#                                       overwrite=overwrite)
#         else:
#             mechanisms = get_mechanisms(job_dir=seed_dir,
#                                         seed=seed,
#                                         scenario=scenario,
#                                         partial_mechanisms=partial_mechanisms,
#                                         constraint_function_power=1,
#                                         classifier_layers=classifier_layers,
#                                         classifier_train_config=classifier_train_config,
#                                         critic=critic,
#                                         mechanism=mechanism,
#                                         train_config=mechanism_train_config,
#                                         from_joint=from_joint,
#                                         overwrite=overwrite)
#
#         results.append(evaluate(seed_dir, mechanisms, is_invertible, marginals, pseudo_oracles, test_dataset,
#                                 overwrite=overwrite))
#
#     print(job_name)
#     print_test_results(results)
#

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
    model = style_gan_model(resolution=128)
    train_config = TrainConfig(32, optax.adam(learning_rate=0.0025, b1=0., b2=.99), 20000, 10, 1000, 1000)
    train_data, test_data = prep_mechanism_data('all', parent_names, True, train_datasets,
                                                test_dataset, batch_size=train_config.batch_size)

    params = train(model=model,
                   job_dir=Path('/tmp/test_style_gan'),
                   seed=8653453,
                   train_data=train_data,
                   test_data=test_data,
                   input_shape=input_shape,
                   optimizer=train_config.optimizer,
                   num_steps=train_config.num_steps,
                   log_every=train_config.log_every,
                   eval_every=train_config.eval_every,
                   save_every=train_config.save_every,
                   overwrite=True,
                   use_jit=True)

    # run_experiment(args.job_dir,
    #                args.data_dir,
    #                args.overwrite,
    #                args.seeds,
    #                baseline=False,
    #                partial_mechanisms=False)
    # for partial_mechanisms in (False, True):
