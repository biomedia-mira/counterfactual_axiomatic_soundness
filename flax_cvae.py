# type: ignore  # Disable mypy type checking because it doesn't work well with Flax.

"""I give no guarantees this will even compile :)"""

import itertools
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import Iterable
from typing import Tuple, Union

from jax import value_and_grad, jit, vmap
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import Optimizer, OptimizerState, Params, ParamsFn
from jax.random import KeyArray
from numpy.typing import NDArray
from tqdm import tqdm
from jax.tree_util import tree_reduce, tree_map
from datasets.confounded_mnist import digit_colour_scenario
from datasets.utils import image_gallery
import jax.random as random

tf.config.experimental.set_visible_devices([], 'GPU')


class Encoder(nn.Module):
    z_dim: int
    hidden_dim: int

    def setup(self):
        n_channels = self.hidden_dim // 4
        self.conv = nn.Sequential(
            [
                nn.Conv(n_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME"),
                nn.relu,
                nn.Conv(n_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME"),
                nn.relu,
            ]
        )
        self.fc = nn.Sequential([nn.Dense(self.hidden_dim), nn.relu])
        self.embed = nn.Sequential([nn.Dense(self.hidden_dim), nn.relu])
        self.z_loc = nn.Dense(self.z_dim)
        self.z_logscale = nn.Dense(self.z_dim)

    def __call__(self, x, y):
        x = self.conv(x)
        x = vmap(lambda _x: _x.flatten())(x)
        x = self.fc(x)
        x = self.embed(jnp.concatenate((x, y), axis=-1))
        return self.z_loc(x), 1e-5 + nn.softplus(self.z_logscale(x))


class Upsample(nn.Module):
    """Upsample module to replicate torch.nn.Upsample. Assumes inputs are in NHWC format
    and only scales H and W dims."""

    scale_factor: int
    method: str = "nearest"

    @nn.compact
    def __call__(self, x):
        new_shape = (
            x.shape[0],
            x.shape[1] * self.scale_factor,
            x.shape[2] * self.scale_factor,
            x.shape[3],
        )
        return jax.image.resize(x, new_shape, method=self.method)


class Decoder(nn.Module):
    hidden_dim: int

    def setup(self):
        n_channels = self.hidden_dim // 4
        self.fc = nn.Sequential(
            [
                nn.Dense(self.hidden_dim),
                nn.relu,
                nn.Dense(n_channels * 7 * 7),
                nn.relu,
            ]
        )
        self.conv = nn.Sequential(
            [
                Upsample(scale_factor=2, method="nearest"),
                nn.Conv(
                    n_channels, kernel_size=(5, 5), strides=(1, 1), padding="SAME"
                ),  # NOTE: not sure if padding SAME is equivalent to fabios code
                nn.relu,
                Upsample(scale_factor=2, method="nearest"),
                nn.Conv(3, kernel_size=(5, 5), strides=(1, 1), padding="SAME"),
            ]
        )
        self.x_loc = nn.Conv(3, kernel_size=(1, 1))
        self.x_logscale = nn.Conv(3, kernel_size=(1, 1))

    def __call__(self, z, y):
        x = jnp.concatenate((z, y), axis=-1)
        x = self.fc(x)
        x = jnp.reshape(x, (x.shape[0], 7, 7, -1))  # NOTE: hardcoded sizes ugh
        x = self.conv(x)
        return nn.sigmoid(self.x_loc(x)), None


class CVAE(nn.Module):
    z_dim: int
    hidden_dim: int = 256

    def setup(self):
        self.encoder = Encoder(z_dim=self.z_dim, hidden_dim=self.hidden_dim)
        self.decoder = Decoder(hidden_dim=self.hidden_dim)

    def __call__(self, rng, x, y):
        qz_loc, qz_scale = self.encoder(x, y)

        z = qz_loc + jnp.exp(qz_scale * 0.5) * jax.random.normal(rng, shape=qz_loc.shape)
        x_loc, _ = self.decoder(z, y)
        log_px = jnp.sum(x * jnp.log(x_loc + 1e-5) + (1 - x) * jnp.log(1 - x_loc + 1e-5), axis=(1, 2, 3))
        kl_qp = 0.5 * jnp.sum(jnp.exp(qz_scale) + qz_loc ** 2 - qz_scale - 1, axis=-1)

        elbo = log_px - kl_qp
        return (
            jnp.mean(-elbo),
            jnp.mean(log_px),
            jnp.mean(kl_qp),
        )  # minimise -ELBO (free energy FE)

    def reconstruct(self, rng, x, y, n_samples=1):
        qz_loc, qz_scale = self.encoder(x, y)
        qz_scale = jnp.exp(0.5 * qz_scale)
        dist = numpyro.distributions.Normal(qz_loc, qz_scale)
        z = jnp.mean(dist.sample(rng, sample_shape=(n_samples,)), axis=0)
        x_loc, _ = self.decoder(z, y)
        return x_loc  # NOTE: no sampling in image/observation space




if __name__ == '__main__':
    job_dir = '/tmp/test_job_flax'
    seed = 32345
    data_dir = Path('/vol/biomedic/users/mm6818/projects/grand_canyon/data')
    train_datasets, test_dataset, parent_dims, _, marginals, input_shape = digit_colour_scenario(data_dir, False, False)
    parent_names = parent_dims.keys()

    batch_size = 512
    train_data = to_numpy_iterator(train_datasets[frozenset()], batch_size, drop_remainder=True)
    test_data = to_numpy_iterator(test_dataset, batch_size, drop_remainder=False)

    #
    model = CVAE(z_dim=16, hidden_dim=256)
    jit_init = jax.jit(model.init)


    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        c_dim = (1, sum(parent_dims.values()))
        jit_init = jax.jit(model.init)
        return jit_init(rng, x=jnp.ones((1, *input_shape[1:])), y=jnp.ones(c_dim), rng=rng)


    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        n_samples = 1
        x, y = inputs
        y = jnp.concatenate([y[key] for key in sorted(y.keys())], axis=-1)
        loss, log_px, kl = model.apply(params, rng, x, y)
        recon = model.apply(params, rng, x, y, n_samples=n_samples, method=model.reconstruct)


        #
        rng, key1, key2 = jax.random.split(rng, 3)
        d = jax.nn.one_hot(
            jax.random.randint(key1, shape=(x.shape[0],), minval=0, maxval=10),
            num_classes=10,
        )
        c = jax.nn.one_hot(
            jax.random.randint(key2, shape=(x.shape[0],), minval=0, maxval=10),
            num_classes=10,
        )
        y = jnp.concatenate((d, c), axis=-1)
        samples = model.apply(params, rng, x, y, n_samples=n_samples, method=model.reconstruct)
        return loss, {'image': x,
                      'recon': recon,
                      'samples': samples,
                      'log_px': log_px,
                      'kl': kl,
                      'elbo': -loss,
                      'loss': loss}


    def init_optimizer_fn(params: Params, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizer

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            (loss, outputs), grads = value_and_grad(apply_fn, has_aux=True)(get_params(opt_state), inputs, rng)
            opt_state = opt_update(i, grads, opt_state)
            grad_norm = jnp.sqrt(tree_reduce(lambda x, y: x + y, tree_map(lambda x: jnp.sum(x ** 2), grads)))
            outputs['grad_norm'] = grad_norm[jnp.newaxis]

            return opt_state, loss, outputs

        return opt_init(params), update, get_params


    params = train(model=(init_fn, apply_fn, init_optimizer_fn),
                   job_dir=Path('/tmp/test_job_flax'),
                   seed=32345,
                   train_data=train_data,
                   test_data=test_data,
                   input_shape=input_shape,
                   optimizer=optimizers.adam(step_size=1e-3),
                   num_steps=10000,
                   log_every=1,
                   eval_every=250,
                   save_every=250)
