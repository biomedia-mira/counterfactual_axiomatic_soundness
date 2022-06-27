from lib2to3.pytree import Leaf
import jax.nn
import math
from posixpath import split
from typing import Callable, List, NamedTuple, Tuple

import jax.lax
import jax.numpy as jnp
import jax.random as random
import jax.scipy
from jax.example_libraries.stax import Conv, Relu, serial, LeakyRelu
from numpyro.distributions import Distribution, Independent, Normal
from jax.nn.initializers import uniform
from staxplus.types import Array, KeyArray, Params, Shape, StaxLayer, is_shape


class TrainableDistribution(NamedTuple):
    init: Callable[[KeyArray, Shape], Tuple[Shape, Params]]
    log_prob: Callable[[Params, Array], Array]
    sample: Callable[[Params, KeyArray, int], Array]

# def standard_normal_prior(event_shape: Shape, variance: float = 1.):
#     prior = Independent(Normal(jnp.zeros(event_shape), variance * jnp.ones(event_shape)), len(event_shape))

#     def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
#         assert input_shape[1:] == event_shape
#         return input_shape, ()

#     def log_prob(params: Params, x: Array) -> Array:
#         return prior.log_prob(x)

#     def sample(params: Params, rng: KeyArray, num_samples: int) -> Array:
#         return prior.sample(rng, (num_samples, ))

#     return TrainableDistribution(init_fn, log_prob, sample)


def learnable_normal_prior() -> TrainableDistribution:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        event_shape = input_shape[1:]
        mu, log_var = jnp.zeros(event_shape), jnp.zeros(event_shape)
        return input_shape, (mu, log_var)

    def prior(params: Params) -> Distribution:
        mu, log_var = params
        scale = jnp.exp(.5 * jax.lax.stop_gradient(log_var)) + 1e-8
        return Independent(Normal(mu, scale), len(mu.shape))

    def log_prob(params: Params, x: Array) -> Array:
        return prior(params).log_prob(x)

    def sample(params: Params, rng: KeyArray, num_samples: int) -> Array:
        return prior(params).sample(rng, (num_samples, ))

    return TrainableDistribution(init_fn, log_prob, sample)


class Bijection(NamedTuple):
    init: Callable[[KeyArray, Shape], Tuple[Shape, Params]]
    forward: Callable[[Params, Array], Tuple[Array, Array]]
    inverse: Callable[[Params, Array], Array]


def chain_bijections(bijections: List[Bijection]) -> Bijection:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        params = []
        for bijection in bijections:
            rng, layer_rng = random.split(rng)
            input_shape, param = bijection.init(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def forward_fn(params: Params, x: Array) -> Tuple[Array, Array]:
        log_prob = 0
        z = x
        for _params, bijection in zip(params, bijections):
            z, log_determinant = bijection.forward(_params, z)
            log_prob = log_prob + log_determinant
        return z, log_prob

    def inverse_fn(params: Params, z: Array) -> Array:
        x = z
        for _params, bijection in reversed(list(zip(params, bijections))):
            x = bijection.inverse(_params, x)
        return x
    return Bijection(init_fn, forward_fn, inverse_fn)


def affine_coupling(network: StaxLayer, flip: bool = False) -> Bijection:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        output_shape, params = network.init(rng, (*input_shape[:-1], input_shape[-1] // 2))
        assert is_shape(output_shape)
        assert output_shape == input_shape
        return output_shape, params

    def shift_and_log_scale_fn(params: Params, x1: Array) -> Tuple[Array, Array]:
        shift, log_scale = jnp.split(network.apply(params, x1), 2, axis=-1)
        return shift, log_scale

    def forward_fn(params: Params, x: Array) -> Tuple[Array, Array]:
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        if flip:
            x2, x1 = x1, x2
        shift, log_scale = shift_and_log_scale_fn(params, x1)
        y2 = x2 * jnp.exp(log_scale) + shift
        if flip:
            x1, y2 = y2, x1
        y = jnp.concatenate([x1, y2], axis=-1)
        axis = tuple(range(1, log_scale.ndim))
        log_determinant = jnp.sum(log_scale, axis=axis)
        return y, log_determinant

    def inverse_fn(params: Params, y: Array) -> Array:
        d = y.shape[-1] // 2
        y1, y2 = y[..., :d], y[..., d:]
        if flip:
            y1, y2 = y2, y1
        shift, log_scale = shift_and_log_scale_fn(params, y1)
        x2 = (y2 - shift) * jnp.exp(-log_scale)
        if flip:
            y1, x2 = x2, y1
        x = jnp.concatenate([y1, x2], axis=-1)
        return x
    return Bijection(init_fn, forward_fn, inverse_fn)


def squeeze_2x():
    """only works on 2D images"""
    log_determinant = jnp.zeros(())

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        assert is_shape(input_shape) and len(input_shape) == 4
        # assert input_shape[-1] % 4 == 0
        output_shape = (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3] * 4)
        return output_shape, ()

    def forward_fn(params: Params, y: Array) -> Tuple[Array, Array]:
        y = jnp.reshape(y, (y.shape[0],
                        y.shape[1] // 2, 2,
                        y.shape[2] // 2, 2,
                        y.shape[-1]))
        y = jnp.transpose(y, (0, 1, 3, 2, 4, 5))
        x = jnp.reshape(y, y.shape[:3] + (4 * y.shape[-1],))
        return x, log_determinant

    def inverse_fn(params: Params, x: Array) -> Array:
        x = jnp.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 2, 2, x.shape[-1] // 4))
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        y = jnp.reshape(x, (x.shape[0],
                            2 * x.shape[1],
                            2 * x.shape[3],
                            x.shape[5]))
        return y

    return Bijection(init_fn, forward_fn, inverse_fn)


def actnorm(eps: float = 1e-8):
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        assert is_shape(input_shape)
        shape = (1, ) * len(input_shape[:-1]) + (input_shape[-1], )
        bias, scale = jnp.zeros(shape=shape), jnp.ones(shape=shape)
        return input_shape, (bias, scale)

    def forward_fn(params: Params, x: Array) -> Tuple[Array, Array]:
        bias, scale = params
        y = (scale + eps) * x + bias
        logdet_factor = y.shape[1] * y.shape[2]
        log_determinant = jnp.sum(jnp.log(jnp.abs(scale))) * logdet_factor
        return y, log_determinant

    def inverse_fn(params: Params, y: Array) -> Array:
        bias, scale = params
        sign = jnp.sign(scale)
        return sign * (y - bias) / (sign * scale + eps)

    return Bijection(init_fn, forward_fn, inverse_fn)


def conv1x1():
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        assert is_shape(input_shape)
        c = int(input_shape[-1])
        w, _ = jnp.linalg.qr(random.normal(rng, shape=(c, c)))
        return input_shape, w

    def conv(img: Array, kernel: Array):
        dimension_numbers = jax.lax.conv_dimension_numbers(img.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))
        return jax.lax.conv_general_dilated(lhs=img,
                                            rhs=kernel,
                                            window_strides=(1, 1),
                                            padding='SAME',
                                            lhs_dilation=(1, 1),
                                            rhs_dilation=(1, 1),
                                            dimension_numbers=dimension_numbers)

    def forward_fn(params: Params, x: Array) -> Tuple[Array, Array]:
        w = params
        _w = jnp.expand_dims(w, axis=(0, 1))
        log_determinant = x.shape[1] * x.shape[2] * jnp.log(jnp.abs(jnp.linalg.det(w)))
        return conv(x, _w), log_determinant

    def inverse_fn(params: Params, y: Array) -> Array:
        w = params
        _w = jnp.expand_dims(jnp.linalg.inv(w), axis=(0, 1))
        return conv(y, _w)

    return Bijection(init_fn, forward_fn, inverse_fn)


def flow(bijection: Bijection, prior: TrainableDistribution) -> TrainableDistribution:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = random.split(rng, 2)
        output_shape, bijection_params = bijection.init(k1, input_shape)
        _, prior_params = prior.init(k2, output_shape)
        return output_shape, (prior_params, bijection_params)

    def log_prob(params: Params, x: Array) -> Array:
        prior_params, bijection_params = params
        z, _log_prob = bijection.forward(bijection_params, x)
        _log_prob = _log_prob + prior.log_prob(prior_params, z)
        return _log_prob

    def sample(params: Params, rng: KeyArray, n_samples: int) -> Array:
        prior_params, bijection_params = params
        z = prior.sample(prior_params, rng, n_samples)
        x = bijection.inverse(bijection_params, z)
        return x
    return TrainableDistribution(init_fn, log_prob, sample)


def split_prior(x1_prior: TrainableDistribution, x2_prior: TrainableDistribution) -> TrainableDistribution:
    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        assert input_shape[-1] % 2 == 0
        k1, k2 = random.split(rng, 2)
        half_shape = (*input_shape[:-1], input_shape[-1] // 2)
        _, x1_prior_params = x1_prior.init(k1, half_shape)
        _, x2_prior_params = x2_prior.init(k2, half_shape)
        return input_shape, (x1_prior_params, x2_prior_params)

    def log_prob(params: Params, x: Array) -> Array:
        x1_prior_params, x2_prior_params = params
        x1, x2 = jnp.split(x, 2, axis=-1)
        log_prob = x1_prior.log_prob(x1_prior_params, x1) + x2_prior.log_prob(x2_prior_params, x2)
        return log_prob

    def sample(params: Params, rng: KeyArray, n_samples: int) -> Array:
        x1_prior_params, x2_prior_params = params
        k1, k2 = random.split(rng, 2)
        x1 = x1_prior.sample(x1_prior_params, k1, n_samples)
        x2 = x2_prior.sample(x2_prior_params, k2, n_samples)
        return jnp.concatenate((x1, x2), axis=-1)

    return TrainableDistribution(init_fn, log_prob, sample)


def _recursive_build_glow(num_levels: int,
                          prior: TrainableDistribution,
                          input_channels: int,
                          depth_per_level: int,
                          flip: bool = False) -> TrainableDistribution:
    if num_levels == 0:
        return prior

    # define network for affine coupling
    network = StaxLayer(
        *serial(
            Conv(32, filter_shape=(3, 3), padding='same'), LeakyRelu,
            Conv(32, filter_shape=(3, 3), padding='same'), LeakyRelu,
            Conv(input_channels * 4, filter_shape=(3, 3), padding='same')
        )
    )

    bijections = [squeeze_2x()]
    for _ in range(depth_per_level):
        bijections += [actnorm(),
                       conv1x1(),
                       affine_coupling(network, flip=flip)]
        flip = not flip
    bijection = chain_bijections(bijections)

    local_prior = learnable_normal_prior()
    prior = split_prior(local_prior, _recursive_build_glow(
        num_levels - 1, prior, input_channels * 2, depth_per_level, flip=flip))
    dist = flow(bijection, prior)
    return dist


def glow(input_shape: Shape, num_levels: int = 3, depth_per_level: int = 3):
    assert len(input_shape) == 4 and input_shape[1] == input_shape[2]
    dist = _recursive_build_glow(num_levels, learnable_normal_prior(), int(input_shape[-1]), depth_per_level)
    return dist
