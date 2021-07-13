from typing import Tuple

import jax
import jax.numpy as jnp
import jax.ops
import jax.random as random
import numpy as np
from jax import jit
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.experimental.stax import Dense, Flatten, Relu, BatchNorm, Sigmoid, Conv, Tanh, Gelu, LeakyRelu, Conv, \
    ConvTranspose

from components.f_divergence import f_divergence
from components.stax_layers import layer_norm, reshape
from trainer.training import ApplyFn, InitFn, InitOptimizerFn, Params, Tree, UpdateFn
from jax.lax import stop_gradient

# discriminative = (Flatten, Dense(1024), BatchNorm(0), Gelu, Dense(512), BatchNorm(0), Gelu)
# generative = (Dense(1024), BatchNorm(0), Gelu, Dense(784), BatchNorm(0), Gelu, Dense(784), reshape((28, 28, 1)), Sigmoid)
# generative = (Dense(1024), BatchNorm(0), Relu, Dense(784 * 64), BatchNorm(0), Relu, reshape((28, 28, 64)),
#               Conv(32, (3, 3), padding='SAME'), BatchNorm(), Relu, Conv(1, (3, 3), padding='SAME'), Sigmoid)
discriminative = (Flatten, Dense(1024), LeakyRelu, Dense(512), LeakyRelu)

generative = (ConvTranspose(64 * 8, filter_shape=(4, 4), padding='VALID'), BatchNorm(), Relu,
              ConvTranspose(64 * 4, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), BatchNorm(), Relu,
              ConvTranspose(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), BatchNorm(), Relu,
              ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), BatchNorm(), Relu,
              ConvTranspose(1, filter_shape=(4, 4), padding='SAME'), Sigmoid)


###

# discriminative = (Flatten, Dense(500), Relu, Dense(250), Relu)
# generative = (Dense(1024), Relu, Dense(784), Relu, Dense(784), reshape((28, 28, 1)), Sigmoid)

# https://github.com/bilal2vec/jax-dcgan/blob/main/dcgan.ipynb

def gan_model() -> Tuple[InitFn, ApplyFn, InitOptimizerFn]:
    divergence_init_fn, divergence_apply_fn = f_divergence(mode='gan', layers=discriminative)
    generator_init_fn, generator_apply_fn = stax.serial(*generative)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]) -> Params:
        _, discriminator_params = divergence_init_fn(rng, (-1, 28, 28, 1))
        _, generator_params = generator_init_fn(rng, (-1, 1, 1, 100))

        return discriminator_params, generator_params

    def apply_fun(params: Params, inputs: Tree[np.ndarray], rng: jnp.ndarray) -> Tuple[jnp.ndarray, Tree[jnp.ndarray]]:
        discriminator_params, generator_params = params
        real_image, _ = inputs
        noise = random.normal(rng, shape=(len(real_image), 1, 1, 100))
        fake_image = generator_apply_fn(generator_params, noise)[:,:28,:28]
        generator_loss = divergence_apply_fn(stop_gradient(discriminator_params), real_image, fake_image)
        discriminator_loss = divergence_apply_fn(discriminator_params, real_image, stop_gradient(fake_image))
        loss = -discriminator_loss + generator_loss
        return loss, {'loss': generator_loss, 'real_image': real_image, 'fake_image': fake_image}

    def init_optimizer_fun(params: Params) -> Tuple[OptimizerState, UpdateFn]:
        opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 0.0002, b1=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Tree[np.ndarray]) \
                -> Tuple[OptimizerState, jnp.ndarray, Tree[jnp.ndarray]]:
            params = get_params(opt_state)
            rng, _ = random.split(random.PRNGKey(i))
            (loss, outputs), grads = jax.value_and_grad(apply_fun, has_aux=True)(params, inputs, rng)
            # grads = (jax.tree_map(lambda x: x * .1, grads[0]), grads[1])

            opt_state = opt_update(i, grads, opt_state)

            return opt_state, loss, outputs

        opt_state = opt_init(params)

        return opt_state, update

    return init_fun, apply_fun, init_optimizer_fun
