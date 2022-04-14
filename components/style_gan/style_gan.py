from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, value_and_grad
from jax.example_libraries.optimizers import Optimizer, OptimizerState, ParamsFn, UpdateFn
from jax.example_libraries.stax import Flatten, LeakyRelu, serial
from jax.lax import stop_gradient

from components import Array, KeyArray, Params, Shape
from components.stax_extension import stax_wrapper
from components.stax_extension import StaxLayer
from components.style_gan.styla_gan_ops import ConvEq, DenseEq, LeakyRelu, mbstddev, normalize_2nd_moment, setup_filter, \
    synthesis_layer, \
    upsample2d


# Generator
def mapping_network(z_dim: int = 512,
                    c_dim: int = 0,
                    w_dim: int = 512,
                    embed_features: Optional[int] = None,
                    layer_features: int = 512,
                    num_layers: int = 8,
                    lr_multiplier: float = 0.01) -> StaxLayer:
    assert z_dim > 0 or c_dim > 0

    init_fn_c, apply_fn_c = DenseEq(w_dim if embed_features is None else embed_features, lr_multiplier=lr_multiplier)
    init_fn_m, apply_fn_m = serial(*[DenseEq(layer_features, lr_multiplier=lr_multiplier), LeakyRelu] * num_layers)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        _, params_c = init_fn_c(k1, (-1, c_dim)) if c_dim > 0 else (None, ())
        output_shape, params_m = init_fn_m(k2, (-1, z_dim + (w_dim if c_dim > 0 else 0)))
        return output_shape, (params_c, params_m)

    def apply_fn(params: Params, z: Array, c: Optional[Array] = None) -> Array:
        params_c, params_m = params
        x = normalize_2nd_moment(z) if z_dim > 0 else None
        if c_dim > 0:
            y = normalize_2nd_moment(apply_fn_c(params_c, c))
            x = jnp.concatenate((x, y), axis=1) if x is not None else y
        return apply_fn_m(params_m, x)

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

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng, num=2)
        output_shape, block_params = block_init_fn(k1, input_shape)
        _, to_rgb_params = to_rgb_init_fn(k2, output_shape)
        return output_shape, (block_params, to_rgb_params)

    def apply_fn(params: Params, inputs: Tuple[Array, Array, Array], rng: KeyArray, **kwargs: Any) \
            -> Tuple[Array, Array, Array]:
        x, y, latent_code = inputs
        block_params, to_rgb_params = params
        x, _ = block_apply_fn(block_params, (x, latent_code), rng=rng)
        if num_layers == 2:
            assert y is not None
            y = upsample2d(y, f=setup_filter((1, 3, 3, 1)), up=2)
        y = to_rgb_apply_fn(to_rgb_params, (x, latent_code), rng=rng)[0] + (y if y is not None else 0)
        # y = jax.image.resize(y, (*x.shape[:-1], y.shape[-1]), method='bilinear') if num_layers == 2 else y
        return x, y, latent_code

    return init_fn, apply_fn


def style_gan_generator(resolution: int,
                        kernel_size: int = 3,
                        use_noise: bool = True,
                        num_image_channels: int = 3,
                        # Mapping Network
                        z_dim: int = 512,
                        c_dim: int = 0,
                        w_dim: int = 512,
                        layer_features: int = 512,
                        num_layers: int = 8,
                        mapping_lr_multiplier: float = 0.01,
                        # Capacity
                        fmap_base: int = 16384,
                        fmap_decay: int = 1,
                        fmap_min: int = 1,
                        fmap_max: int = 512,
                        fmap_const: Optional[int] = None) -> StaxLayer:
    def nf(stage: int) -> int:
        return int(jnp.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max))

    mapping_init_fn, mapping_apply_fn = mapping_network(z_dim=z_dim,
                                                        c_dim=c_dim,
                                                        w_dim=w_dim,
                                                        layer_features=layer_features,
                                                        num_layers=num_layers,
                                                        lr_multiplier=mapping_lr_multiplier)

    resolution_log2 = int(jnp.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    # Generator
    generator_blocks = [synthesis_block(num_layers=1 if res == 2 else 2,
                                        up=res != 2,
                                        latent_dim=layer_features,
                                        fmaps_out=nf(res - 1),
                                        kernel_size=kernel_size,
                                        use_noise=use_noise,
                                        num_image_channels=num_image_channels)
                        for res in range(2, resolution_log2 + 1)]

    _gen_init_fn, _gen_apply_fn = serial(*generator_blocks)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2, k3 = jax.random.split(rng, num=3)
        _, mapping_params = mapping_init_fn(k1, ())
        const_input = jax.random.normal(k2, (1, 4, 4, fmap_const if fmap_const is not None else nf(1)))
        output_shape, gen_params = _gen_init_fn(k3, const_input.shape)
        return output_shape, (mapping_params, const_input, gen_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray, **kwargs: Any) -> Array:
        z, c = inputs if c_dim > 0 else (inputs, None)
        mapping_params, const_input, gen_params = params
        latent = mapping_apply_fn(mapping_params, z, c)
        x = jnp.repeat(const_input, z.shape[0], axis=0)
        y = None
        x, y, latent = _gen_apply_fn(gen_params, (x, y, latent), rng=rng)
        return y

    return init_fn, apply_fn


# Discriminator
def discriminator_block(fmaps: Tuple[int, int], use_residual: bool = False) -> StaxLayer:
    path_init_fn, path_apply_fn = serial(
        *(ConvEq(fmaps[0], filter_shape=(3, 3)), LeakyRelu,
          ConvEq(fmaps[1], filter_shape=(3, 3), down=True)), LeakyRelu)
    res_init_fn, res_apply_fn = ConvEq(fmaps[1], filter_shape=(1, 1), down=True)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        output_shape, path_params = path_init_fn(k1, input_shape)
        _, residual_params = res_init_fn(k2, input_shape)
        return output_shape, (path_params, residual_params)

    def apply_fn(params: Params, inputs: Array, **kwargs: Any) -> Array:
        path_params, residual_params = params
        output = path_apply_fn(path_params, inputs, **kwargs)
        if use_residual:
            output = (output + res_apply_fn(residual_params, inputs)) * jnp.sqrt(.5)
        return output

    return init_fn, apply_fn


def style_gan_discriminator(resolution: int,
                            c_dim: int = 0,
                            # Capacity
                            fmap_base: int = 16384,
                            fmap_decay: int = 1,
                            fmap_min: int = 1,
                            fmap_max: int = 512,
                            fmap_const: Optional[int] = None,
                            # Discriminator
                            mbstd_group_size: int = None,  # None
                            mbstd_num_features: int = 1,
                            num_mapping_layers: int = 0,
                            mapping_lr_multiplier: float = .1) -> StaxLayer:
    def nf(stage: int) -> int:
        return int(jnp.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max))

    resolution_log2 = int(jnp.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4

    mapping_fmaps = nf(0)
    disc_c_init_fn, disc_c_apply_fn = serial(DenseEq(mapping_fmaps, mapping_lr_multiplier),
                                             stax_wrapper(normalize_2nd_moment),
                                             *[DenseEq(mapping_fmaps, mapping_lr_multiplier)] * num_mapping_layers)

    discriminator_blocks = \
        [ConvEq(nf(resolution_log2 - 1), filter_shape=(1, 1)), LeakyRelu,
         *[discriminator_block((nf(res - 1), nf(res - 2))) for res in range(resolution_log2, 2, -1)],
         mbstddev(mbstd_num_features, mbstd_group_size),
         ConvEq(nf(1), filter_shape=(1, 1)), LeakyRelu, Flatten, DenseEq(nf(0)), LeakyRelu,
         DenseEq(1 if c_dim == 0 else mapping_fmaps)]

    disc_init_fn, disc_apply_fn = serial(*discriminator_blocks)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        _, disc_c_params = disc_c_init_fn(k1, (input_shape[0], c_dim)) if c_dim > 0 else (None, ())
        output_shape, disc_params = disc_init_fn(k2, input_shape)
        return output_shape, (disc_params, disc_c_params)

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Array:
        z, c = inputs if c_dim > 0 else (inputs, None)
        disc_params, disc_c_params = params
        x = disc_apply_fn(disc_params, z)
        if c_dim > 0:
            c = disc_c_apply_fn(disc_c_params, c)
            x = jnp.sum(x * c, axis=1, keepdims=True) / jnp.sqrt(mapping_fmaps)
        return x

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
              mapping_lr_multiplier: float = 0.01,
              # Capacity
              fmap_base: int = 16384,
              fmap_decay: int = 1,
              fmap_min: int = 1,
              fmap_max: int = 512,
              fmap_const: Optional[int] = None):
    generator_init, generator_apply_fn = style_gan_generator(resolution)
    discriminator_init_fn, discriminator_apply_fn = style_gan_discriminator(resolution)

    def init_fn(rng: KeyArray, input_shape: Shape) -> Tuple[Shape, Params]:
        k1, k2 = jax.random.split(rng)
        _, generator_params = generator_init(k1, ())
        output_shape, discriminator_params = discriminator_init_fn(k2, input_shape)
        return output_shape, (generator_params, discriminator_params)

    def apply_fn(params: Params, inputs: Any, rng: KeyArray):
        inputs = inputs[0]
        generator_params, discriminator_params = params
        z = random.normal(rng, (inputs.shape[0], z_dim))
        fake_image = generator_apply_fn(generator_params, z, rng)
        fake_logits = discriminator_apply_fn(discriminator_params, fake_image)
        real_logits = discriminator_apply_fn(discriminator_params, inputs)
        loss_fake = jax.nn.softplus(fake_logits)
        loss_real = jax.nn.softplus(-real_logits)
        loss = jnp.mean(loss_fake + loss_real)
        return loss, {'fake_image': fake_image, 'loss_real': loss_real, 'loss_fake': loss_fake,
                            'loss': loss[jnp.newaxis]}

    def step_generator(params: Params, inputs: Any, rng: KeyArray):
        inputs = inputs[0]
        generator_params, discriminator_params = params
        z = random.normal(rng, (inputs.shape[0], z_dim))
        fake_image = generator_apply_fn(generator_params, z, rng)
        fake_logits = discriminator_apply_fn(stop_gradient(discriminator_params), inputs)
        loss = jnp.mean(jax.nn.softplus(-fake_logits))
        return loss, {'fake_image': fake_image, 'gen_loss': loss[jnp.newaxis]}

    # @jit
    # def regularise_generator(params, inputs):
    #     pl_grads = jax.grad(lambda *args: jnp.sum(generator_apply_fn(*args) * pl_noise), argnums=1)(params, inputs)[-1]
    #     pl_lengths = jnp.sqrt(jnp.mean(jnp.sum(jnp.square(pl_grads), axis=2), axis=1))
    #     pl_mean_new = pl_mean + config.pl_decay * (jnp.mean(pl_lengths) - pl_mean)
    #     pl_penalty = jnp.square(pl_lengths - pl_mean_new) * config.pl_weight
    #     loss = jnp.mean(pl_penalty) * config.G_reg_interval
    #
    #     return loss, pl_mean_new

    def step_discriminator(params: Params, inputs: Any, rng: KeyArray):
        inputs = inputs[0]
        generator_params, discriminator_params = params
        z = random.normal(rng, (inputs.shape[0], z_dim))
        fake_image = generator_apply_fn(stop_gradient(generator_params), z, rng)
        fake_logits = discriminator_apply_fn(discriminator_params, fake_image)
        real_logits = discriminator_apply_fn(discriminator_params, inputs)
        loss_fake = jax.nn.softplus(fake_logits)
        loss_real = jax.nn.softplus(-real_logits)
        loss = jnp.mean(loss_fake + loss_real)
        return loss, {'loss_real': loss_real, 'loss_fake': loss_fake, 'loss': loss[jnp.newaxis]}

    def init_optimizer_fn(params: Params, optimizer: Optimizer) -> Tuple[OptimizerState, UpdateFn, ParamsFn]:
        opt_init, opt_update, get_params = optimizer

        @jit
        def update(i: int, opt_state: OptimizerState, inputs: Any, rng: KeyArray) -> Tuple[OptimizerState, Array, Any]:
            k1, k2 = random.split(rng, 2)
            (disc_loss, disc_outputs), disc_grads \
                = value_and_grad(step_discriminator, has_aux=True)(get_params(opt_state), inputs=inputs, rng=k1)
            opt_state = opt_update(i, disc_grads, opt_state)
            (gen_loss, gen_outputs), gen_grads \
                = value_and_grad(step_generator, has_aux=True)(get_params(opt_state), inputs=inputs, rng=k2)
            opt_state = opt_update(i, gen_grads, opt_state)

            return opt_state, disc_loss + gen_loss, {**disc_outputs, **gen_outputs}

        return opt_init(params), update, get_params

    return init_fn, apply_fn, init_optimizer_fn
