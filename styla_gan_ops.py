import jax
import jax.lax
import jax.numpy as jnp

dimension_numbers = ('NHWC', 'HWIO', 'NHWC')


def minibatch_stddev_layer(x, group_size=None, num_new_features=1):
    if group_size is None:
        group_size = x.shape[0]
    else:
        # Minibatch must be divisible by (or smaller than) group_size.
        group_size = min(group_size, x.shape[0])

    G = group_size
    F = num_new_features
    _, H, W, C = x.shape
    c = C // F

    # [NHWC] Cast to FP32.
    y = x.astype(jnp.float32)
    # [GnHWFc] Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y = jnp.reshape(y, newshape=(G, -1, H, W, F, c))
    # [GnHWFc] Subtract mean over group.
    y -= jnp.mean(y, axis=0)
    # [nHWFc] Calc variance over group.
    y = jnp.mean(jnp.square(y), axis=0)
    # [nHWFc] Calc stddev over group.
    y = jnp.sqrt(y + 1e-8)
    # [nF] Take average over channels and pixels.
    y = jnp.mean(y, axis=(1, 2, 4))
    # [nF] Cast back to original data type.
    y = y.astype(x.dtype)
    # [n11F] Add missing dimensions.
    y = jnp.reshape(y, newshape=(-1, 1, 1, F))
    # [NHWC] Replicate over group and pixels.
    y = jnp.tile(y, (G, H, W, 1))
    return jnp.concatenate((x, y), axis=3)


def normalize_2nd_moment(x, eps=1e-8):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=1, keepdims=True) + eps)


def setup_filter(f, normalize=True, flip_filter=False, gain=1, separable=None):
    """
    Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f (tensor): Tensor or python list of the shape.
        normalize (bool): Normalize the filter so that it retains the magnitude.
                          for constant input signal (DC)? (default: True).
        flip_filter (bool): Flip the filter? (default: False).
        gain (int): Overall scaling factor for signal magnitude (default: 1).
        separable: Return a separable filter? (default: select automatically).

    Returns:
        (tensor): Output filter of shape [filter_height, filter_width] or [filter_taps]
    """
    # Validate.
    if f is None:
        f = 1
    f = jnp.array(f, dtype=jnp.float32)
    assert f.ndim in [0, 1, 2]
    assert f.size > 0
    if f.ndim == 0:
        f = f[jnp.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.size >= 8)
    if f.ndim == 1 and not separable:
        f = jnp.outer(f, f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= jnp.sum(f)
    if flip_filter:
        for i in range(f.ndim):
            f = jnp.flip(f, axis=i)
    f = f * (gain ** (f.ndim / 2))
    return f


def upfirdn2d(x, f, padding=(2, 1, 2, 1), up=1, down=1, strides=(1, 1), flip_filter=False, gain=1):
    if f is None:
        f = jnp.ones((1, 1), dtype=jnp.float32)

    B, H, W, C = x.shape
    padx0, padx1, pady0, pady1 = padding

    # upsample by inserting zeros
    x = jnp.reshape(x, newshape=(B, H, 1, W, 1, C))
    x = jnp.pad(x, pad_width=((0, 0), (0, 0), (0, up - 1), (0, 0), (0, up - 1), (0, 0)))
    x = jnp.reshape(x, newshape=(B, H * up, W * up, C))

    # padding
    x = jnp.pad(x, pad_width=((0, 0), (max(pady0, 0), max(pady1, 0)), (max(padx0, 0), max(padx1, 0)), (0, 0)))
    x = x[:, max(-pady0, 0): x.shape[1] - max(-pady1, 0), max(-padx0, 0): x.shape[2] - max(-padx1, 0)]

    # setup filter
    f = f * (gain ** (f.ndim / 2))
    if not flip_filter:
        for i in range(f.ndim):
            f = jnp.flip(f, axis=i)

    # convole filter
    f = jnp.repeat(jnp.expand_dims(f, axis=(-2, -1)), repeats=C, axis=-1)
    if f.ndim == 4:
        x = jax.lax.conv_general_dilated(x,
                                         f.astype(x.dtype),
                                         window_strides=strides or (1,) * (x.ndim - 2),
                                         padding='valid',
                                         feature_group_count=C,
                                         dimension_numbers=dimension_numbers)
    else:
        x = jax.lax.conv_general_dilated(x,
                                         jnp.expand_dims(f, axis=0).astype(x.dtype),
                                         window_strides=strides or (1,) * (x.ndim - 2),
                                         padding='valid',
                                         feature_group_count=C,
                                         dimension_numbers=dimension_numbers)
        x = jax.lax.conv_general_dilated(x,
                                         jnp.expand_dims(f, axis=1).astype(x.dtype),
                                         window_strides=strides or (1,) * (x.ndim - 2),
                                         padding='valid',
                                         feature_group_count=C,
                                         dimension_numbers=dimension_numbers)
    x = x[:, ::down, ::down]
    return x


def upsample_conv_2d(x, w, k=None, factor=2, gain=1, padding=0):
    """
    Fused upsample convolution.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x (tensor): Input tensor of the shape [N, H, W, C].
        w (tensor): Weight tensor of the shape [filterH, filterW, inChannels, outChannels].
                    Grouped convolution can be performed by inChannels = x.shape[0] // numGroups.
        k (tensor): FIR filter of the shape [firH, firW] or [firN].
                    The default is [1] * factor, which corresponds to nearest-neighbor upsampling.
        factor (int): Integer upsampling factor (default: 2).
        gain (float): Scaling factor for signal magnitude (default: 1.0).
        padding (int): Number of pixels to pad or crop the output on each side (default: 0).

    Returns:
        (tensor): Output of the shape [N, H * factor, W * factor, C].
    """
    assert isinstance(factor, int) and factor >= 1
    assert isinstance(padding, int)

    # Check weight shape.
    ch, cw, _inC, _outC = w.shape
    inC = w.shape[2]
    outC = w.shape[3]
    assert cw == ch

    # Fast path for 1x1 convolution.
    if cw == 1 and ch == 1:
        x = jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), padding='VALID',
                                         dimension_numbers=dimension_numbers)
        k = setup_filter(k, gain=gain * (factor ** 2))
        pad0 = (k.shape[0] + factor - cw) // 2 + padding
        pad1 = (k.shape[0] - factor) // 2 + padding
        x = upfirdn2d(x, f=k, up=factor, padding=(pad0, pad1, pad0, pad1))
        return x

    # Setup filter kernel.
    k = setup_filter(k, gain=gain * (factor ** 2))
    assert k.shape[0] == k.shape[1]

    # Determine data dimensions.
    stride = (factor, factor)
    output_shape = ((x.shape[1] - 1) * factor + ch, (x.shape[2] - 1) * factor + cw)
    num_groups = x.shape[3] // inC

    # Transpose weights.
    w = jnp.reshape(w, (ch, cw, inC, num_groups, -1))
    w = jnp.transpose(w[::-1, ::-1], (0, 1, 4, 3, 2))
    w = jnp.reshape(w, (ch, cw, -1, num_groups * inC))
    # Execute.
    x = jax.lax.conv_transpose(lhs=x, rhs=w, strides=stride, padding='VALID', dimension_numbers=dimension_numbers)

    pad0 = (k.shape[0] + factor - cw) // 2 + padding
    pad1 = (k.shape[0] - factor - cw + 3) // 2 + padding
    x = upfirdn2d(x=x, f=k, padding=(pad0, pad1, pad0, pad1))
    return x


def conv2d(x, w, up=False, resample_kernel=None, padding=0):
    kernel = w.shape[0]
    assert w.shape[1] == kernel
    assert kernel >= 1 and kernel % 2 == 1

    num_groups = x.shape[3] // w.shape[2]

    w = w.astype(x.dtype)
    if up:
        x = upsample_conv_2d(x, w, k=resample_kernel, padding=padding)
    else:
        padding_mode = {0: 'SAME', -(kernel // 2): 'VALID'}[padding]
        x = jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), padding=padding_mode,
                                         feature_group_count=num_groups, dimension_numbers=dimension_numbers)
    return x


def modulated_conv2d_layer(x, w, s, fmaps, kernel, up=False, demodulate=True, resample_kernel=None,
                           fused_modconv=False):
    assert kernel >= 1 and kernel % 2 == 1

    ww = w[jnp.newaxis]  # [BkkIO] Introduce minibatch dimension.
    # Modulate.
    ww *= s[:, jnp.newaxis, jnp.newaxis, :, jnp.newaxis].astype(w.dtype)  # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = jax.lax.rsqrt(jnp.sum(jnp.square(ww), axis=(1, 2, 3)) + 1e-8)  # [BO] Scaling factor.
        ww *= d[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x, (1, -1, x.shape[2], x.shape[3]))  # Fused => reshape minibatch to convolution groups.
        x = jnp.transpose(x, axes=(0, 2, 3, 1))
        w = jnp.reshape(jnp.transpose(ww, (1, 2, 3, 0, 4)), (ww.shape[1], ww.shape[2], ww.shape[3], -1))
    else:
        x *= s[:, jnp.newaxis, jnp.newaxis].astype(x.dtype)  # [BIhw] Not fused => scale input activations.

    # 2D convolution.
    x = conv2d(x, w.astype(x.dtype), up=up, resample_kernel=resample_kernel)

    # Reshape/scale output.
    if fused_modconv:
        x = jnp.transpose(x, axes=(0, 3, 1, 2))
        x = jnp.reshape(x,
                        (-1, fmaps, x.shape[2], x.shape[3]))  # Fused => reshape convolution groups back to minibatch.
        x = jnp.transpose(x, axes=(0, 2, 3, 1))
    elif demodulate:
        x *= d[:, jnp.newaxis, jnp.newaxis].astype(x.dtype)  # [BOhw] Not fused => scale output activations.

    return x

