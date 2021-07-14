from typing import Tuple

from jax.experimental.stax import Conv, Dense, Relu, Sigmoid, BatchNorm

from components.stax_layers import layer_norm, conv_residual_block, unet
from components.linear_attention import linear_attention_layer


def get_layers_rgb_mode(input_shape: Tuple[int, ...]):
    assert len(input_shape) == 4
    _, h, w, input_dim = input_shape

    discriminative = (Conv(32, filter_shape=(3, 3), padding='VALID'), Relu,
                      Conv(16, filter_shape=(3, 3), padding='VALID'), Relu,
                      Conv(8, filter_shape=(3, 3), padding='VALID'), Relu)

    # generative = (Conv(32, filter_shape=(3, 3), padding='SAME'), Relu, BatchNorm(),
    #               Conv(64, filter_shape=(3, 3), padding='SAME'), Relu, BatchNorm(),
    #               Conv(128, filter_shape=(3, 3), padding='SAME'), Relu, BatchNorm(),
    #               Conv(input_dim, filter_shape=(3, 3), padding='SAME'), Sigmoid)

    generative = (Conv(32, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(64, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(128, filter_shape=(3, 3), padding='SAME'), Relu,
                  Conv(input_dim, filter_shape=(3, 3), padding='SAME'), Sigmoid)

    return discriminative, discriminative, generative


def get_layers_jpeg_mode(input_shape: Tuple[int, ...]):
    assert len(input_shape) == 3
    _, seq_len, seq_dim = input_shape
    discriminative = (layer_norm(-1), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Relu,
                      layer_norm(-1), Dense(seq_dim // 2), Relu,
                      layer_norm(-1), Dense(seq_dim // 2), Relu)

    generative = (layer_norm(-1), Dense(10), Relu,
                  layer_norm(-1), linear_attention_layer(10, seq_len, heads=10), Relu,
                  layer_norm(-1), Dense(seq_dim), Relu,
                  layer_norm(-1), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Relu,
                  layer_norm(-1), Dense(seq_dim), Relu,
                  layer_norm(-1), linear_attention_layer(seq_dim, seq_len, heads=seq_dim), Relu, Dense(seq_dim))
    return discriminative, discriminative, generative


def get_layers(mode: str, input_shape: Tuple[int, ...]):
    if mode == 'rgb':
        return get_layers_rgb_mode(input_shape)
    elif mode == 'jpeg':
        return get_layers_jpeg_mode(input_shape)
    else:
        raise ValueError(f'mode {mode} is not supported!')
