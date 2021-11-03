import shutil
from pathlib import Path
from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.ops
import jax.random as random
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import jit
from jax.experimental import optimizers
from jax.experimental.optimizers import OptimizerState, l2_norm

from components.f_gan import f_gan
from components.typing import Array, Params, PRNGKey, Shape
from model.train import Model, train
from style_gan import style_gan


def gan(resolution: int, num_image_channels: int, z_dim: int = 64, c_dim=0) -> Model:
    generator, discriminator = style_gan(resolution=resolution, num_image_channels=num_image_channels, z_dim=z_dim,
                                         c_dim=c_dim, fmap_max=128, fmap_min=1, layer_features=128, num_layers=2,
                                         use_noise=True)
    divergence_init_fn, divergence_apply_fn = f_gan(mode='wasserstein', layers=[discriminator], trick_g=True,
                                                    disc_penalty=0.)
    generator_init_fn, generator_apply_fn = generator

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Params:
        k1, k2 = jax.random.split(rng)
        _, discriminator_params = divergence_init_fn(k1, input_shape)
        _, generator_params = generator_init_fn(k2, (input_shape[0], z_dim))
        return discriminator_params, generator_params

    def apply_fn(params: Params, inputs: Any, rng: PRNGKey) -> Tuple[Array, Any]:
        discriminator_params, generator_params = params
        real_image, labels = inputs
        k1, k2 = jax.random.split(rng)
        noise = random.normal(k1, shape=(len(real_image), z_dim))
        fake_image = generator_apply_fn(generator_params, noise, k2)

        loss, disc_loss, gen_loss = divergence_apply_fn(discriminator_params, real_image, fake_image)
        return loss, {'loss': loss[jnp.newaxis],
                      'gen_loss': gen_loss[jnp.newaxis],
                      'disc_loss': disc_loss[jnp.newaxis],
                      'real_image': real_image, 'fake_image': fake_image}

    def init_optimizer_fn(params: Params):
        #
        # opt_init_d, opt_update_d, get_params_d = optimizers.adam(step_size=lambda x: 0.001, b1=0.0, b2=0.9)
        # opt_init_g, opt_update_g, get_params_g = optimizers.adam(step_size=lambda x: 0.001, b1=0.0, b2=0.9)
        #
        # def get_params(opt_state):
        #     return (get_params_d(opt_state[0]), get_params_g(opt_state[1]))
        #
        # @jit
        # def update(i: int, opt_state: Tuple[OptimizerState, OptimizerState], inputs: Any, rng: PRNGKey) -> Tuple[OptimizerState, Array, Any]:
        #     opt_state_d, opt_state_g = opt_state
        #
        #     (loss, outputs), grads = jax.value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
        #     opt_state_g = opt_update_g(i, grads[1], opt_state_g)
        #     opt_state = (opt_state_d, opt_state_g)
        #     for _ in range(1):
        #         (loss, outputs), grads = jax.value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
        #         opt_state_d = opt_update_d(i, grads[0], opt_state_d)
        #         opt_state = (opt_state_d, opt_state_g)
        #
        #     return opt_state, loss, outputs
        #
        # init_opt_state = (opt_init_d(params[0]), opt_init_g(params[1]))
        #
        # return init_opt_state, update, get_params

        #
        opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 0.002, b1=0.0, b2=0.9)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: Array) -> Tuple[OptimizerState, Array, Any]:
            params = get_params(opt_state)
            (loss, outputs), grads = jax.value_and_grad(apply_fn, has_aux=True)(params, inputs, rng)
            opt_state = opt_update(i, grads, opt_state)

            return opt_state, loss, outputs

        opt_state = opt_init(params)

        return opt_state, update, get_params

    return init_fn, apply_fn, init_optimizer_fn


def encode(image: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = (tf.image.resize(tf.cast(image, tf.float32), (32, 32)) - tf.constant(127.5)) / tf.constant(127.5)
    labels = tf.one_hot(labels, 10)
    return image, labels


if __name__ == '__main__':

    overwrite = True
    job_dir = Path('/tmp/gan_test')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)
    batch_size = 128
    num_image_channels = 1
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised=True,
                                  data_dir='/tmp')
    ds_train = ds_train.map(encode)

    train_data = tfds.as_numpy(ds_train.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))

    model = gan(resolution=32, num_image_channels=num_image_channels)

    model_path = job_dir / 'model.npy'
    params = train(model=model,
                   input_shape=(batch_size, 32, 32, num_image_channels),
                   job_dir=job_dir,
                   num_steps=100000,
                   train_data=train_data,
                   test_data=None,
                   log_every=1,
                   eval_every=500,
                   save_every=500)
