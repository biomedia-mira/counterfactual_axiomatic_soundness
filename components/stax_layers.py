from typing import Tuple, Union

import jax
import jax.experimental.stax as stax
from jax.experimental.stax import Conv, Relu
from jax.nn import normalize
import jax.numpy as jnp
import jax.random as random


# differentiable rounding operation
def differentiable_round(x):
    return x - (jax.lax.stop_gradient(x) - jnp.round(x))


def layer_norm(axis: Union[int, Tuple[int, ...]]):
    axis = axis if isinstance(axis, tuple) else tuple((axis,))

    def init_fun(rng, input_shape):
        features_shape = tuple(s if i in axis else 1 for i, s in enumerate(input_shape))
        bias = jnp.zeros(shape=features_shape)
        scale = jnp.ones(shape=features_shape)
        return input_shape, (bias, scale)

    def apply_fun(params, inputs, **kwargs):
        bias, scale = params
        return scale * normalize(inputs, axis=axis) + bias

    return init_fun, apply_fun


def reshape(output_shape):
    def init_fun(rng, input_shape):
        return (-1, *output_shape), ()

    def apply_fun(params, inputs, **kwargs):
        return jnp.reshape(inputs, (-1, *output_shape))

    return init_fun, apply_fun


def conv_residual_block(in_channels: int, out_channels: int, filter_shape: Tuple[int, int] = (3, 3)):
    residual = out_channels % in_channels == 0 or in_channels % out_channels == 0
    init_fun, net_apply_fun = stax.serial(*(layer_norm(-1), Relu, Conv(out_channels, filter_shape, padding='SAME')))

    def apply_fun(params, inputs, **kwargs):
        outputs = net_apply_fun(params, inputs)
        if not residual:
            return outputs
        if out_channels >= in_channels:
            return outputs + jnp.repeat(inputs, repeats=out_channels // in_channels, axis=-1)
        else:
            return inputs[..., :out_channels] + outputs

    return init_fun, apply_fun


def unet(channels: Tuple[int, ...] = (64, 128, 256, 512, 1024)):
    encoder, decoder = [], []
    for in_c, out_c in zip(channels[:-1], channels[1:]):
        encoder.append(stax.serial(conv_residual_block(in_c, out_c), conv_residual_block(out_c, out_c)))
        decoder.append(stax.serial(conv_residual_block(out_c + in_c, in_c), conv_residual_block(in_c, in_c)))

    def init_fun(rng, input_shape):
        u_net_params = []
        for (encoder_init_fn, _), (decoder_init_fn, _), in_c in zip(encoder, decoder, channels):
            (_, _, _, out_c), enc_params = encoder_init_fn(rng, (-1, -1, -1, in_c))
            _, dec_params = decoder_init_fn(rng, (-1, -1, -1, in_c + out_c))
            u_net_params.append((enc_params, dec_params))

        return input_shape, u_net_params

    def apply_fun(params, inputs, **kwargs):
        outputs, shapes = [], []
        x = inputs
        for (_, _apply_fun), (_params, _) in zip(encoder, params):
            outputs.append(x), shapes.append(x.shape)
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]), method='bilinear')
            x = _apply_fun(_params, x)

        for (_, _apply_fun), (_, _params), output, shape in zip(decoder[::-1], params[::-1], outputs[::-1],
                                                                shapes[::-1]):
            x = jax.image.resize(x, shape=(*shape[:-1], x.shape[-1]), method='bilinear')
            x = _apply_fun(_params, jnp.concatenate((x, output), axis=-1))
        return x

    return init_fun, apply_fun
