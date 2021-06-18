import functools
import operator as op
from typing import List, Tuple, Callable, Any, Dict

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, partial
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, GeneralConv, LeakyRelu
from jax.scipy.ndimage import map_coordinates

Conv = functools.partial(GeneralConv, ('NCHW', 'HWIO', 'NCHW'))


def apply_fun_n_times(f: Callable, n: int) -> Any:
    return lambda x: x if n == 0 else apply_fun_n_times(f, n - 1)(f(x))


def cumprod(array):
    return functools.reduce(op.mul, array, 1)


# KL estimator
def kl_estimator(layer_sizes: Tuple[int, ...] = (100, 100)):
    layers = []
    for layer_size in layer_sizes:
        layers.append(Dense(layer_size))
        layers.append(Relu)
    layers.append(Dense(1))
    net_init_fun, net_apply_fun = stax.serial(*layers)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        return net_init_fun(rng, (input_shape[0], cumprod(input_shape[1:])))

    @jit
    def apply_fun(params: List[Tuple[jnp.ndarray, ...]], p_sample: jnp.ndarray, q_sample: jnp.ndarray):
        t_joint = net_apply_fun(params, jnp.reshape(p_sample, (p_sample.shape[0], -1)))
        t_marginals = net_apply_fun(params, jnp.reshape(q_sample, (q_sample.shape[0], -1)))
        return jnp.mean(t_joint) - jnp.log(jnp.mean(jnp.exp(t_marginals)))

    return init_fun, apply_fun


# Mutual Information estimator -MINE
def mi_estimator(layer_sizes: Tuple[int, ...] = (100, 100)):
    kl_init_fun, kl_apply_fun = kl_estimator(layer_sizes)

    def init_fun(rng: jnp.ndarray, input_1_shape: Tuple[int, ...], input_2_shape: Tuple[int, ...]):
        assert input_1_shape[0] == input_2_shape[0]
        return kl_init_fun(rng, (input_1_shape[0], cumprod(input_1_shape[1:]) + cumprod(input_2_shape[1:])))

    def input_fun(x1: jnp.ndarray, x2: jnp.ndarray):
        return jnp.concatenate((jnp.reshape(x1, (x1.shape[0], -1)), jnp.reshape(x2, (x2.shape[0], -1))), axis=-1)

    def apply_fun(params: List[Tuple[jnp.ndarray, ...]],
                  joint_dist_samples: Tuple[jnp.ndarray, jnp.ndarray],
                  marginal_dist_samples: Tuple[jnp.ndarray, jnp.ndarray]):
        return kl_apply_fun(params, input_fun(*joint_dist_samples), input_fun(*marginal_dist_samples))

    return init_fun, apply_fun


def warp(x: jnp.ndarray, displacement: jnp.ndarray) -> jnp.ndarray:
    shape = x.shape[2:]
    grid = jnp.meshgrid(*[jnp.linspace(0, shape[i] - 1, shape[i]) for i in range(x.ndim - 2)], indexing='ij')
    warped_grid = jnp.repeat(jnp.expand_dims(jnp.stack(grid) + displacement, axis=1), x.shape[1], axis=1)
    return vmap(vmap(partial(map_coordinates, order=1), in_axes=0), in_axes=0)(x, warped_grid)


# mechanism that acts on image based on categorical parent variable
def mechanism(parent_name: str,
              parent_dims: Dict[str, int],
              output_channels: int = 3,
              num_channels: Tuple[int, ...] = (40, 40, 40)):
    dim = sum(parent_dims.values()) + parent_dims[parent_name]
    layers = []
    for channels in num_channels:
        layers.append(Conv(channels, filter_shape=(3, 3), padding='SAME'))
        layers.append(LeakyRelu)
    layers.append(Conv(output_channels + 2, filter_shape=(3, 3), padding='SAME'))
    net_init_fun, net_apply_fun = stax.serial(*layers)

    def init_fun(rng: jnp.ndarray, input_shape: Tuple[int, ...]):
        return net_init_fun(rng, (input_shape[0], input_shape[1] + dim, *input_shape[2:]))

    @jit
    def apply_fun(params: List[Tuple[jnp.ndarray, ...]],
                  image: jnp.ndarray,
                  parents: Dict[str, jnp.ndarray],
                  do_parent: jnp.ndarray):
        def broadcast(parent: jnp.ndarray) -> jnp.ndarray:
            parent = apply_fun_n_times(lambda x: x[..., np.newaxis], image.ndim - parent.ndim)(parent)
            return jnp.broadcast_to(parent, (image.shape[0], parent.shape[1], *image.shape[2:]))

        net_inputs = jnp.concatenate((image, broadcast(jnp.concatenate([*parents.values(), do_parent], axis=1))),
                                     axis=1)
        net_outputs = net_apply_fun(params, net_inputs)
        output, disp = net_outputs[:, :output_channels], net_outputs[:, output_channels:]
        return output

    return init_fun, apply_fun
