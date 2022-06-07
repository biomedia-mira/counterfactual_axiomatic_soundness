from staxplus.conditional_vae import c_vae
from staxplus.f_gan import f_gan
from staxplus.layers import broadcast_together, layer_norm, reshape, resize, ResBlock
from staxplus.train import train
from staxplus.types import (Array, ArrayTree, GradientTransformation, KeyArray, Model, OptState, Params, Shape,
                            ShapeTree, StaxLayer)

Reshape = reshape
Resize = resize
BroadcastTogether = broadcast_together
LayerNorm2D = layer_norm(axis=(1, 2, 3))
LayerNorm1D = layer_norm(axis=(1,))
PixelNorm2D = layer_norm(axis=(3,))

__all__ = ['Array',
           'ArrayTree',
           'BroadcastTogether',
           'c_vae',
           'f_gan',
           'GradientTransformation',
           'KeyArray',
           'LayerNorm1D',
           'LayerNorm2D',
           'Model',
           'OptState',
           'Params',
           'ResBlock',
           'Reshape',
           'Resize',
           'Shape',
           'ShapeTree',
           'StaxLayer',
           'train']
