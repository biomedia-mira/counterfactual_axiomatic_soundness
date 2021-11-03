import functools
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.stax import Conv, Dense, elementwise, Flatten, serial
from jax.nn.initializers import normal, zeros, he_normal,he_uniform

from components.stax_extension import stax_wrapper
from components.typing import Array, Params, PRNGKey, Shape, StaxLayer, StaxLayerConstructor
from spectral_norm import estimate_spectral_norm
from styla_gan_ops import minibatch_stddev_layer, modulated_conv2d_layer, normalize_2nd_moment

mod_conv = partial(modulated_conv2d_layer, fused_modconv=False, resample_kernel=(1, 3, 3, 1))


def leaky_relu(x: Array, negative_slope: Array = 1e-2, gain: float = 1.4142135623730951) -> Array:
    return 1.0 * jax.nn.leaky_relu(x, negative_slope)


LeakyRelu = elementwise(leaky_relu)
_w_init = he_uniform() #normal(stddev=1.)
_b_init = zeros


def eq_params(w: Params, b: Params, lr_multiplier: float) -> Params:
    return w * lr_multiplier / np.sqrt(np.prod(w.shape[:-1])), b * lr_multiplier


def calc_spectral_norm(f, params, input_shape):
    w, b = params
    return estimate_spectral_norm(lambda x: f(params, x) - b, input_shape)


def spectral_norm_wrapper(layer_constructor: StaxLayerConstructor) -> StaxLayerConstructor:
    @functools.wraps(layer_constructor)
    def _layer_constructor(*args: Any, **kwargs: Any) -> StaxLayer:
        init_fn, apply_fn = layer_constructor(*args, **kwargs)

        @functools.wraps(apply_fn)
        def _apply_fn(params: Params, inputs: Array, **_kwargs: Any) -> Array:
            w, b = params
            norm = calc_spectral_norm(apply_fn, params, inputs.shape)
            return apply_fn((w / norm, b), inputs, **_kwargs)

        return init_fn, _apply_fn

    return _layer_constructor


def equalize_lr_params(layer_constructor: StaxLayerConstructor) -> StaxLayerConstructor:
    def _layer_constructor(*args: Any, lr_multiplier: float = 1., **kwargs: Any) -> StaxLayer:
        init_fn, apply_fn = layer_constructor(*args, **kwargs)

        def _init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
            output_shape, params = init_fn(rng, input_shape)
            return output_shape, jax.tree_map(lambda x: x / lr_multiplier, params)

        def _apply_fn(params: Params, inputs: Array, **_kwargs: Any) -> Array:
            w, b = params
            return apply_fn(eq_params(w, b, lr_multiplier), inputs, **_kwargs)

        return _init_fn, _apply_fn

    return _layer_constructor


def mbstddev(num_new_features: int, group_size: Optional[int] = None) -> StaxLayer:
    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        return (*input_shape[:-1], input_shape[-1] + num_new_features), ()

    def apply_fn(params, inputs, **kwargs) -> Array:
        return minibatch_stddev_layer(inputs, group_size, num_new_features)

    return init_fn, apply_fn


def mapping_network(z_dim: int = 512,
                    c_dim: int = 0,
                    w_dim: int = 512,
                    layer_features: int = 512,
                    num_layers: int = 8,
                    lr_multiplier: float = 0.01) -> StaxLayer:
    assert z_dim > 0 or c_dim > 0
    # DenseEq = equalize_lr_params(partial(Dense, W_init=normal(stddev=stddev), b_init=zeros))
    DenseEq = equalize_lr_params(partial(Dense, W_init=_w_init, b_init=_b_init))

    init_fn_c, apply_fn_c = DenseEq(w_dim, lr_multiplier=lr_multiplier)
    init_fn_m, apply_fn_m = serial(
        *[DenseEq(layer_features, lr_multiplier=lr_multiplier), LeakyRelu] * num_layers)

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        _, params_c = init_fn_c(k1, (-1, c_dim)) if c_dim > 0 else (None, ())
        output_shape, params_m = init_fn_m(k2, (-1, z_dim + (w_dim if c_dim > 0 else 0)))
        return output_shape, (params_c, params_m)

    def apply_fn(params: Params, z: Array, c: Optional[Array] = None) -> Array:
        params_c, params_m = params
        x = normalize_2nd_moment(z) if z_dim > 0 else None
        y = normalize_2nd_moment(apply_fn_c(params_c, c)) if c_dim > 0 else None
        x_cat_y = x if c_dim <= 0 else (y if z_dim <= 0 else jnp.concatenate((x, y), axis=1))
        return apply_fn_m(params_m, x_cat_y)

    return init_fn, apply_fn


def synthesis_layer(latent_dim: int,
                    fmaps_out: int,
                    kernel_size: int,
                    demodulate: bool = True,
                    up: bool = False,
                    use_noise: bool = True,
                    activation: Callable[[Array], Array] = leaky_relu,
                    w_init: Callable[[PRNGKey, Shape], Array] = _w_init,
                    lr_multiplier: float = 1.) -> StaxLayer:
    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2, k3, k4 = jax.random.split(rng, num=4)
        fmaps_in = input_shape[-1]
        w_a, b_a = w_init(k1, (latent_dim, fmaps_in)) / lr_multiplier, jnp.ones(input_shape[-1]) / lr_multiplier
        w_c, b_c = w_init(k3, (kernel_size, kernel_size, fmaps_in, fmaps_out)) / lr_multiplier, _b_init(k4, (fmaps_out, ))
        noise_gain = jnp.zeros(())
        output_shape = (input_shape[0], input_shape[1] * (2 if up else 1), input_shape[2] * (2 if up else 1), fmaps_out)
        return output_shape, (w_a, b_a, w_c, b_c, noise_gain)

    def apply_fn(params: Params, inputs: Tuple[Array, Array], rng: PRNGKey, **kwargs: Any) -> Tuple[Array, Array]:
        x, latent_code = inputs
        w_a, b_a, w_c, b_c, noise_gain = params
        (w_a, b_a), (w_c, b_c) = eq_params(w_a, b_a, lr_multiplier), eq_params(w_c, b_c, lr_multiplier)
        style = jnp.dot(latent_code, w_a) + b_a
        x = mod_conv(x, w_c, style, fmaps_out, kernel_size, up, demodulate) + b_c
        noise_shape = (x.shape[0], x.shape[1], x.shape[2], 1)
        noise = noise_gain * jax.random.normal(rng, shape=noise_shape) if use_noise else jnp.zeros(noise_shape)
        return activation(x + noise), latent_code

    return init_fn, apply_fn


def synthesis_block(num_layers: int,
                    up: bool,
                    latent_dim: int,
                    fmaps_out: int,
                    kernel_size: int,
                    use_noise: bool = True,
                    num_image_channels: int = 3) -> StaxLayer:
    block_init_fn, block_apply_fn = \
        serial(*[synthesis_layer(latent_dim=latent_dim,
                                 fmaps_out=fmaps_out,
                                 kernel_size=kernel_size,
                                 up=i == 0 and up,
                                 use_noise=use_noise)
                 for i in range(num_layers)])
    to_rgb_init_fn, to_rgb_apply_fn = synthesis_layer(latent_dim=latent_dim,
                                                      fmaps_out=num_image_channels,
                                                      kernel_size=1,
                                                      demodulate=False,
                                                      up=False,
                                                      use_noise=False,
                                                      activation=lambda x: x)

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng, num=2)
        output_shape, block_params = block_init_fn(k1, input_shape)
        _, to_rgb_params = to_rgb_init_fn(k2, output_shape)
        return output_shape, (block_params, to_rgb_params)

    def apply_fn(params: Params, inputs: Tuple[Array, Array, Array], rng: PRNGKey, **kwargs: Any) \
            -> Tuple[Array, Array, Array]:
        x, y, latent_code = inputs
        block_params, to_rgb_params = params
        x, _ = block_apply_fn(block_params, (x, latent_code), rng=rng)
        y = jax.image.resize(y, (*x.shape[:-1], y.shape[-1]), method='bilinear') if up else y
        y = to_rgb_apply_fn(to_rgb_params, (x, latent_code), rng=rng)[0] + y
        return x, y, latent_code

    return init_fn, apply_fn


# ConvSN = spectral_norm_wrapper(equalize_lr_params(partial(Conv, W_init=_w_init, b_init=_b_init)))
# DenseSN = spectral_norm_wrapper(equalize_lr_params(partial(Dense, W_init=_w_init, b_init=_b_init)))

ConvSN = equalize_lr_params(partial(Conv, W_init=_w_init, b_init=_b_init))
DenseSN = equalize_lr_params(partial(Dense, W_init=_w_init, b_init=_b_init))


def discriminator_block(fmaps: Tuple[int, int]) -> StaxLayer:
    path_init_fn, path_apply_fn = serial(
        *(ConvSN(fmaps[0], filter_shape=(3, 3), padding='SAME'), LeakyRelu,
          ConvSN(fmaps[1], filter_shape=(2, 2), strides=(2, 2))), LeakyRelu)
    res_init_fn, res_apply_fn = ConvSN(fmaps[1], filter_shape=(2, 2), strides=(2, 2))

    def init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        output_shape, path_params = path_init_fn(k1, input_shape)
        _, residual_params = res_init_fn(k2, input_shape)
        return output_shape, (path_params, residual_params)

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        path_params, residual_params = params
        return path_apply_fn(path_params, inputs, **kwargs) + res_apply_fn(residual_params, inputs)

    return init_fn, apply_fn


def style_gan(resolution: int,
              kernel_size: int = 3,
              use_noise: bool = True,
              num_image_channels: int = 3,
              # Mapping Network
              z_dim: int = 512,
              c_dim: int = 0,
              w_dim: int = 512,
              layer_features: int = 512,
              num_layers: int = 8,
              # Capacity
              fmap_base: int = 16384,
              fmap_decay: int = 1,
              fmap_min: int = 1,
              fmap_max: int = 512,
              fmap_const: Optional[int] = None,
              # Discriminator
              mbstd_group_size: int = None,
              mbstd_num_features: int = 1
              ):
    def nf(stage: int) -> int:
        return int(jnp.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max))

    mapping_init_fn, mapping_apply_fn = mapping_network(z_dim, c_dim, w_dim, layer_features, num_layers)

    resolution_log2 = int(jnp.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    generator_blocks = [synthesis_block(num_layers=1 if res == 2 else 2,
                                        up=res != 2,
                                        latent_dim=layer_features,
                                        fmaps_out=nf(res - 1),
                                        kernel_size=kernel_size,
                                        use_noise=use_noise,
                                        num_image_channels=num_image_channels)
                        for res in range(2, resolution_log2 + 1)]

    gen_init_fn, gen_apply_fn = serial(*generator_blocks)

    def generator_init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2, k3 = jax.random.split(rng, num=3)
        _, mapping_params = mapping_init_fn(k1, ())
        const_input = jax.random.normal(k2, (1, 4, 4, fmap_const if fmap_const is not None else nf(1)))
        output_shape, gen_params = gen_init_fn(k3, const_input.shape)
        return output_shape, (mapping_params, const_input, gen_params)

    def generator_apply_fn(params: Params, inputs: Any, rng: PRNGKey, **kwargs: Any) -> Array:
        z, c = inputs if c_dim > 0 else (inputs, None)
        mapping_params, const_input, gen_params = params
        latent = mapping_apply_fn(mapping_params, z, c)
        x = jnp.repeat(const_input, z.shape[0], axis=0)
        y = jnp.zeros((*x.shape[:-1], num_image_channels))
        x, y, latent = gen_apply_fn(gen_params, (x, y, latent), rng=rng)
        return y

    ##
    mapping_fmaps = nf(0)
    num_mapping_layers = 0
    disc_c_init_fn, disc_c_apply_fn = \
        serial(
            *[DenseSN(mapping_fmaps), stax_wrapper(normalize_2nd_moment),
              *[DenseSN(mapping_fmaps)] * num_mapping_layers])

    discriminator_blocks = \
        [ConvSN(nf(resolution_log2 - 1), filter_shape=(1, 1)), LeakyRelu,
         *[discriminator_block((nf(res - 1), nf(res - 2))) for res in range(resolution_log2, 2, -1)],
         mbstddev(mbstd_num_features, mbstd_group_size),
         ConvSN(nf(1), filter_shape=(1, 1)), LeakyRelu, Flatten, DenseSN(nf(0)), LeakyRelu,
         DenseSN(1 if c_dim == 0 else mapping_fmaps)]

    disc_init_fn, disc_apply_fn = serial(*discriminator_blocks)

    def discriminator_init_fn(rng: PRNGKey, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        output_shape, disc_params = disc_init_fn(k1, input_shape)
        _, disc_c_params = disc_c_init_fn(k2, (input_shape[0], c_dim)) if c_dim > 0 else (None, ())
        return output_shape, (disc_params, disc_c_params)

    def discriminator_apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Array:
        z, c = inputs if c_dim > 0 else (inputs, None)
        disc_params, disc_c_params = params
        x = disc_apply_fn(disc_params, z)
        if c_dim > 0:
            c = disc_c_apply_fn(disc_c_params, c)
            x = jnp.sum(x * c, axis=1, keepdims=True) / jnp.sqrt(mapping_fmaps)
        return x

    return (generator_init_fn, generator_apply_fn), (discriminator_init_fn, discriminator_apply_fn)
