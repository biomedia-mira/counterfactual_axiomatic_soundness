from typing import Tuple

import jax.nn
import jax.numpy as jnp
from jax.experimental.stax import Conv, Dense, Relu, Sigmoid, BatchNorm, elementwise, ConvTranspose

from components.stax_layers import layer_norm, conv_residual_block, unet
from components.linear_attention import linear_attention_layer
from components.mlp_mixer import mixer_layer


def jpeg_sigmoid(x):
    return jnp.concatenate((jax.nn.tanh(x[..., :3]), x[..., 3:]), axis=-1)


def get_layers_rgb_mode(input_shape: Tuple[int, ...]):
    assert len(input_shape) == 4
    _, h, w, input_dim = input_shape

    discriminative = (Conv(32, filter_shape=(3, 3), padding='VALID'), Relu,
                      Conv(16, filter_shape=(3, 3), padding='VALID'), Relu,
                      Conv(8, filter_shape=(3, 3), padding='VALID'), Relu)

    # generative = (Conv(16, filter_shape=(3, 3), padding='VALID'), Relu,
    #               Conv(32, filter_shape=(3, 3), padding='VALID'), Relu,
    #               Conv(64, filter_shape=(3, 3), padding='VALID'), Relu,
    #               Conv(128, filter_shape=(3, 3), padding='VALID'), Relu,
    #               ConvTranspose(64, filter_shape=(3, 3)), Relu,
    #               ConvTranspose(32, filter_shape=(3, 3)), Relu,
    #               ConvTranspose(16, filter_shape=(3, 3)), Relu,
    #               ConvTranspose(input_dim, filter_shape=(3, 3)), Sigmoid)

    # This one works for color at least
    generative = (Conv(32, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(64, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(128, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(64, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(32, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(input_dim, filter_shape=(3, 3), padding='SAME'), Sigmoid)

    return discriminative, discriminative, generative


def get_layers_jpeg_mode(input_shape: Tuple[int, ...]):
    assert len(input_shape) == 3
    _, seq_len, seq_dim = input_shape
    discriminative = (mixer_layer(seq_len, 16), Relu,
                      mixer_layer(seq_len, 16), Relu)

    generative = (mixer_layer(seq_len, 16), Relu,
                  mixer_layer(seq_len, 32), Relu,
                  mixer_layer(seq_len, 32), Relu,
                  mixer_layer(seq_len, 16), Relu,
                  mixer_layer(seq_len, seq_dim), Relu,
                  elementwise(jpeg_sigmoid))

    return discriminative, discriminative, generative


def get_layers(mode: str, input_shape: Tuple[int, ...]):
    if mode == 'rgb':
        return get_layers_rgb_mode(input_shape)
    elif mode == 'jpeg':
        return get_layers_jpeg_mode(input_shape)
    else:
        raise ValueError(f'mode {mode} is not supported!')
