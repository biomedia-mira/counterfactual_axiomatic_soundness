import shutil
from pathlib import Path
from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.ops
import jax.random as random
from jax import jit
from jax.experimental import optimizers, stax
from jax.experimental.optimizers import OptimizerState
from jax.experimental.stax import BatchNorm, Conv, ConvTranspose, Relu, Sigmoid, leaky_relu, elementwise, Tanh
from jax.nn.initializers import normal as normal_init

from components.f_gan import f_gan
from components.typing import Array, Shape
from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.train import Params, train, Model

from functools import partial


def differentiable_round(x):
    return x - (jax.lax.stop_gradient(x) - jnp.round(x))


def act(x):
    return x - (jax.lax.stop_gradient(x - jnp.tanh(x)))

Act = elementwise(act)
LeakyRelu = elementwise(partial(leaky_relu, negative_slope=.2))
# discriminator_layers = (Conv(64, filter_shape=(4, 4), padding='VALID'), BatchNorm(gamma_init=), LeakyRelu,
#                         Conv(64 * 2, filter_shape=(4, 4), padding='VALID'), BatchNorm(), LeakyRelu,
#                         Conv(64 * 4, filter_shape=(4, 4), padding='VALID'), BatchNorm(), LeakyRelu,
#                         Conv(64 * 8, filter_shape=(4, 4), padding='VALID'), BatchNorm(), LeakyRelu,
#                         Conv(1, filter_shape=(4, 4), padding='VALID'), BatchNorm(), LeakyRelu)

# generator_layers = (ConvTranspose(64 * 8, filter_shape=(4, 4), padding='VALID'), BatchNorm(), Relu,
#                     ConvTranspose(64 * 4, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), BatchNorm(), Relu,
#                     ConvTranspose(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), BatchNorm(), Relu,
#                     ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), BatchNorm(), Relu,
#                     ConvTranspose(3, filter_shape=(4, 4), strides=(1, 1), padding='SAME'), Sigmoid)

##
from components.stax_extension import LayerNorm2D, layer_norm


# Norm = BatchNorm(gamma_init=normal_init(0.02))
# Norm = layer_norm(axis=-1, scale_init=normal_init(0.02))

def norm():
    def init_fun(rng, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return inputs

    return init_fun, apply_fun

init_var = 0.02
init_var = 0.02
Norm = norm()
Norm = layer_norm(axis=(1, 2, 3), scale_init=normal_init(0.02))
Norm = BatchNorm(gamma_init=normal_init(init_var))
Norm = layer_norm(axis=(1,2,3), scale_init=normal_init(0.02))

discriminator_layers = (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=normal_init(init_var))
                        ,Norm, LeakyRelu,
                        Conv(64 * 2, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=normal_init(init_var))
                        ,Norm, LeakyRelu,
                        Conv(1, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=normal_init(init_var)))
generator_layers = (
    ConvTranspose(64 * 4, filter_shape=(3, 3), strides=(2, 2), padding='VALID', W_init=normal_init(init_var)),
    Norm, LeakyRelu,
    ConvTranspose(64 * 4, filter_shape=(4, 4), strides=(1, 1), padding='VALID', W_init=normal_init(init_var)),
    Norm, LeakyRelu,
    ConvTranspose(64 * 2, filter_shape=(3, 3), strides=(2, 2), padding='VALID', W_init=normal_init(init_var)),
    Norm, LeakyRelu,
    ConvTranspose(3, filter_shape=(4, 4), strides=(2, 2), padding='VALID', W_init=normal_init(init_var)), Sigmoid
)


def gan(noise_shape: Shape = (1, 1, 128)) -> Model:
    divergence_init_fn, divergence_apply_fn = f_gan(mode='gan', layers=discriminator_layers, trick_g=True)
    generator_init_fn, generator_apply_fn = stax.serial(*generator_layers)

    def init_fn(rng: Array, input_shape: Shape) -> Params:
        _, discriminator_params = divergence_init_fn(rng, input_shape)
        _, generator_params = generator_init_fn(rng, (-1, *noise_shape))

        return discriminator_params, generator_params

    @jit
    def apply_fn(params: Params, inputs: Any, rng: Array) -> Tuple[Array, Any]:
        discriminator_params, generator_params = params
        real_image, _ = inputs[frozenset()]
        noise = random.normal(rng, shape=(len(real_image), *noise_shape))
        fake_image = generator_apply_fn(generator_params, noise)[:, :28, :28, :]
        divergence, disc_loss, gen_loss = divergence_apply_fn(discriminator_params, real_image, fake_image)
        loss = gen_loss - disc_loss
        return loss, {'divergence': divergence[jnp.newaxis],
                      'gen_loss': gen_loss[jnp.newaxis],
                      'disc_loss': disc_loss[jnp.newaxis],
                      'real_image': real_image, 'fake_image': fake_image}

    def init_optimizer_fn(params: Params):
        opt_init, opt_update, get_params = optimizers.adam(step_size=lambda x: 1e-4, b1=0.5)

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: Array) -> Tuple[OptimizerState, Array, Any]:
            params = get_params(opt_state)
            (loss, outputs), grads = jax.value_and_grad(apply_fn, has_aux=True)(params, inputs, rng)
            opt_state = opt_update(i, grads, opt_state)
            return opt_state, loss, outputs

        opt_state = opt_init(params)

        return opt_state, update, get_params

    return init_fn, apply_fn, init_optimizer_fn


if __name__ == '__main__':

    overwrite = True
    job_dir = Path('/tmp/gan_test')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)

    train_data, test_data, parent_dims, marginals, input_shape = create_confounded_mnist_dataset(batch_size=2048,
                                                                                                 debug=False)
    print('Loaded dataset...')

    model = gan()

    model_path = job_dir / 'model.npy'
    params = train(model=model,
                   input_shape=input_shape,
                   job_dir=job_dir,
                   num_steps=100000,
                   train_data=train_data,
                   test_data=test_data,
                   log_every=1,
                   eval_every=500,
                   save_every=500)
