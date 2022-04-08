from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import tensorflow as tf
from jax import jit, value_and_grad, vmap
from jax.example_libraries import optimizers
from jax.example_libraries.optimizers import Optimizer, OptimizerState, Params, ParamsFn, UpdateFn
from jax.example_libraries.stax import Conv, Dense, elementwise, FanInConcat, FanOut, Flatten, parallel, Relu, serial, \
    Sigmoid
from jax.image import resize
from jax.nn import softplus
from jax.random import KeyArray
from jax.tree_util import tree_map, tree_reduce
from datasets.confounded_mnist import digit_colour_scenario
from train import Array, Model, Shape, StaxLayer, to_numpy_iterator, train

tf.config.experimental.set_visible_devices([], 'GPU')


def stax_wrapper(fn: Callable[[Array], Array]) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return input_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return fn(inputs)

    return init_fn, apply_fn


def reshape(output_shape: Shape) -> StaxLayer:
    def init_fun(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return output_shape, ()

    def apply_fun(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return jnp.reshape(inputs, output_shape)

    return init_fun, apply_fun


def up_sample(new_shape: Shape) -> StaxLayer:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        return new_shape, ()

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        return vmap(partial(resize, shape=new_shape[1:], method='nearest'))(inputs)

    return init_fn, apply_fn


Reshape = reshape
UpSample = up_sample


def vae_loss(x, x_loc, qz_loc, qz_logvar) -> Tuple[Array, Array, Array, Array]:
    log_px = jnp.sum(x * jnp.log(x_loc + 1e-5) + (1 - x) * jnp.log(1 - x_loc + 1e-5), axis=(1, 2, 3))
    kl_qp = 0.5 * jnp.sum(jnp.exp(qz_logvar) + qz_loc ** 2 - qz_logvar - 1, axis=-1)
    elbo = log_px - kl_qp
    loss = jnp.mean(-elbo)
    return loss, elbo, log_px, kl_qp


# Stax
def standard_vae(parent_dims: Dict[str, int],
                 latent_dim: int = 16,
                 hidden_dim: int = 128) -> Model:
    """ Implements VAE with independent normal posterior, standard normal prior and, standard normal likelihood (l2)"""
    assert len(parent_dims) > 0
    n_channels = hidden_dim // 4

    encoder_layers = (
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Relu,
        Conv(n_channels, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), Relu,
        Flatten, Dense(hidden_dim), Relu)
    decoder_layers = (
        Dense(hidden_dim), Relu, Dense(n_channels * 7 * 7),
        Relu,
        Reshape((-1, 7, 7, n_channels)), UpSample((-1, 14, 14, n_channels)),
        Conv(n_channels, filter_shape=(5, 5), strides=(1, 1), padding='SAME'), Relu,
        UpSample((-1, 28, 28, n_channels)),
        Conv(3, filter_shape=(5, 5), strides=(1, 1), padding='SAME'),
        Conv(3, filter_shape=(1, 1), strides=(1, 1), padding='SAME'), Sigmoid)

    enc_init_fn, enc_apply_fn = serial(parallel(serial(*encoder_layers), stax_wrapper(lambda x: x)),
                                       FanInConcat(axis=-1), Dense(hidden_dim), Relu,
                                       Dense(hidden_dim), Relu,
                                       FanOut(2),
                                       parallel(Dense(latent_dim), serial(Dense(latent_dim), elementwise(softplus))))
    dec_init_fn, dec_apply_fn = serial(FanInConcat(axis=-1), *decoder_layers)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = random.split(rng, 2)
        c_dim = sum(parent_dims.values())
        (enc_output_shape, _), enc_params = enc_init_fn(k1, (input_shape, (-1, c_dim)))
        output_shape, dec_params = dec_init_fn(k2, (enc_output_shape, (-1, c_dim)))
        return output_shape, (enc_params, dec_params)

    def reconstruct(params: Params, x: Array, y: Array, rng: KeyArray) -> Tuple[Array, Array, Array, Array]:
        enc_params, dec_params = params
        qz_loc, qz_logvar = enc_apply_fn(enc_params, (x, y))
        z = qz_loc + jnp.exp(qz_logvar * 0.5) * jax.random.normal(rng, shape=qz_loc.shape)
        x_loc = dec_apply_fn(dec_params, (z, y))
        return x_loc, qz_loc, qz_logvar, z

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        x, parents = inputs
        y = jnp.concatenate([parents[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)

        x_loc, qz_loc, qz_logvar, z = reconstruct(params, x, y, rng)
        loss, elbo, log_px, kl_qp = vae_loss(x, x_loc, qz_loc, qz_logvar)

        # conditional samples
        random_parents_1 = {'colour': parents['colour'], 'digit': parents['colour']}
        y_1 = jnp.concatenate([random_parents_1[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        samples_1, _, _, _ = reconstruct(params, x, y_1, rng)

        random_parents_2 \
            = {
            p_name: jax.nn.one_hot(jax.random.randint(_rng, shape=(x.shape[0],), minval=0, maxval=10), num_classes=10)
            for _rng, p_name in zip(random.split(rng, len(parent_dims)), parent_dims.keys())}
        y_2 = jnp.concatenate([random_parents_2[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        samples_2, _, _, _ = reconstruct(params, x, y_2, rng)

        return loss, {'image': x,
                      'recon': x_loc,
                      'log_px': log_px,
                      'kl': kl_qp,
                      'elbo': elbo,
                      'samples_1': samples_1,
                      'samples_2': samples_2,
                      'variance': jnp.mean(jnp.exp(qz_logvar), axis=-1),
                      'mean': jnp.mean(qz_loc, axis=-1),
                      'snr': jnp.abs(jnp.mean(qz_loc / jnp.exp(qz_logvar), axis=-1)),
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

    return init_fn, apply_fn, init_optimizer_fn


# Flax
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


def flax_cvae_wrapper(latent_dim, hidden_dim):
    model = CVAE(z_dim=latent_dim, hidden_dim=hidden_dim)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        c_dim = (1, sum(parent_dims.values()))
        jit_init = jax.jit(model.init)
        return (), jit_init(rng, x=jnp.ones((1, *input_shape[1:])), y=jnp.ones(c_dim), rng=rng)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray) -> Tuple[Array, Dict[str, Array]]:
        n_samples = 1
        x, parents = inputs
        y = jnp.concatenate([parents[key] for key in sorted(parents.keys())], axis=-1)
        loss, log_px, kl = model.apply(params, rng, x, y)
        recon = model.apply(params, rng, x, y, n_samples=n_samples, method=model.reconstruct)

        ##
        # conditional samples
        random_parents_1 = {'colour': parents['colour'], 'digit': parents['colour']}
        y_1 = jnp.concatenate([random_parents_1[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        samples_1 = model.apply(params, rng, x, y_1, n_samples=n_samples, method=model.reconstruct)

        random_parents_2 \
            = {
            p_name: jax.nn.one_hot(jax.random.randint(_rng, shape=(x.shape[0],), minval=0, maxval=10), num_classes=10)
            for _rng, p_name in zip(random.split(rng, len(parent_dims)), parent_dims.keys())}
        y_2 = jnp.concatenate([random_parents_2[parent_name] for parent_name in sorted(parent_dims.keys())], axis=-1)
        samples_2 = model.apply(params, rng, x, y_2, n_samples=n_samples, method=model.reconstruct)

        return loss, {'image': x,
                      'recon': recon,
                      'samples_1': samples_1,
                      'samples_2': samples_2,
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

    return init_fn, apply_fn, init_optimizer_fn


if __name__ == '__main__':
    data_dir = Path('/vol/biomedic/users/mm6818/projects/grand_canyon/data')
    train_datasets, test_dataset, parent_dims, _, marginals, input_shape = digit_colour_scenario(data_dir, False, False)
    parent_names = parent_dims.keys()

    batch_size = 512
    train_data = to_numpy_iterator(train_datasets[frozenset()], batch_size, drop_remainder=True)
    test_data = to_numpy_iterator(test_dataset, batch_size, drop_remainder=False)

    model = standard_vae(parent_dims, latent_dim=16, hidden_dim=256)
    params = train(model=model,
                   job_dir=Path('/tmp/test_job_stax'),
                   seed=32345,
                   train_data=train_data,
                   test_data=test_data,
                   input_shape=input_shape,
                   optimizer=optimizers.adam(step_size=1e-3),
                   num_steps=10000,
                   log_every=1,
                   eval_every=250,
                   save_every=250)

    model = flax_cvae_wrapper(latent_dim=16, hidden_dim=256)
    params = train(model=model,
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
