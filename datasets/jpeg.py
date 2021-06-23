from typing import Tuple, List

import numpy as np
import tensorflow as tf
import itertools
from functools import partial

from scipy import misc
import imageio
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

t_luma = [16, 11, 10, 16, 24, 40, 51, 61,
          12, 12, 14, 19, 26, 58, 60, 55,
          14, 13, 16, 24, 40, 57, 69, 56,
          14, 17, 22, 29, 51, 87, 80, 62,
          18, 22, 37, 56, 68, 109, 103, 77,
          24, 35, 55, 64, 81, 104, 113, 92,
          49, 64, 78, 87, 103, 121, 120, 101,
          72, 92, 95, 98, 112, 100, 103, 99]

t_chroma = [17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99]


def zigzag(rows: int, columns: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    order: List[List[int]] = [[] for i in range(rows + columns - 1)]
    for i in range(columns):
        for j in range(rows):
            sum_ = i + j
            if sum_ % 2 == 0:
                order[sum_].insert(0, i * rows + j)  # add at beginning
            else:
                order[sum_].append(i * rows + j)  # add at end of the list
    zigzag_order = tuple(itertools.chain(*order))
    reverse_zigzag_order = tuple(sorted(range(rows * columns), key=lambda i: zigzag_order[i]))
    return zigzag_order, reverse_zigzag_order


def get_t_luma(quality: float) -> np.ndarray:
    assert 0 < quality <= 100
    s = lambda q: 5000 / q if q < 50 else 200 - 2 * q
    return np.floor((s(quality) * np.array(t_luma) + 50) / 100)


def dct_2d(image: tf.Tensor) -> tf.Tensor:
    def _dct(x: tf.Tensor) -> tf.Tensor:
        return tf.transpose(tf.signal.dct(x, norm='ortho'), perm=(0, 1, 2, 4, 3))

    return _dct(_dct(image))


def idct_2d(image: tf.Tensor) -> tf.Tensor:
    def _idct(x: tf.Tensor) -> tf.Tensor:
        return tf.signal.idct(tf.transpose(x, perm=(0, 1, 2, 4, 3)), norm='ortho')

    return _idct(_idct(image))


def dense_to_seq(dct_img: tf.Tensor) -> tf.Tensor:
    def _dense_to_seq(x: tf.Tensor) -> tf.Tensor:
        return tf.concat((tf.cast(tf.where(x != 0), dtype=tf.float32), x[x != 0][..., tf.newaxis]), axis=-1)

    return tf.map_fn(_dense_to_seq, dct_img, fn_output_signature=tf.RaggedTensorSpec((None, 4), ragged_rank=0))


def seq_to_dense(seq: List[tf.Tensor], img_shape: tf.TensorShape) -> tf.Tensor:
    def _seq_to_dense(x: tf.Tensor) -> tf.Tensor:
        return tf.sparse.to_dense(
            tf.SparseTensor(dense_shape=img_shape, indices=tf.cast(x[..., :-1], tf.int64), values=x[..., -1]))

    return tf.map_fn(_seq_to_dense, seq, fn_output_signature=tf.TensorSpec(shape=img_shape))


def get_jpeg_encode_decode_fns(max_seq_len: int, block_size: Tuple[int, int] = (8, 8), quality: float = 50.,
                               chroma_subsample: bool = True):
    assert block_size == (8, 8)
    bs = block_size[0] * block_size[1]
    zigzag_order, reverse_zigzag_order = zigzag(*block_size)
    t_luma_tf = tf.reshape(tf.convert_to_tensor(get_t_luma(quality), dtype=tf.float32), (1, 1, 1, 8, 8))
    t_chroma_tf = tf.reshape(tf.convert_to_tensor(t_chroma, dtype=tf.float32), (1, 1, 1, 8, 8))

    c = 2 if chroma_subsample else 1
    chroma_slope = tf.convert_to_tensor([1, c, c, 1], dtype=tf.float32)
    u_offset = tf.convert_to_tensor([bs, 0, 0, 1], dtype=tf.float32)
    v_offset = tf.convert_to_tensor([2 * bs, 0, 0, 1], dtype=tf.float32)

    def get_padding(dim_len: int, block_len: int) -> Tuple[int, int]:
        block_len = block_len * 2 if chroma_subsample else block_len
        total_pad = int((dim_len // block_len + 1) * block_len - dim_len)
        return total_pad // 2, total_pad // 2 + total_pad % 2

    def pad(image: tf.Tensor) -> tf.Tensor:
        paddings = ((0, 0), *(get_padding(s, b) for s, b in zip(image.shape[1:-1], block_size)), (0, 0))
        return tf.pad(image, paddings=paddings, mode='REFLECT')

    def dct_quantization(component: tf.Tensor, quantization_matrix: tf.Tensor) -> tf.Tensor:
        sizes = (1, *block_size, 1)
        patches = tf.image.extract_patches(component, sizes=sizes, strides=sizes, rates=(1, 1, 1, 1), padding='SAME')
        patches = tf.reshape(patches, shape=(*patches.shape[:-1], *block_size))
        dct_blocks = dct_2d(patches)
        dct_blocks = tf.round(dct_blocks / quantization_matrix)
        dct_img = tf.gather(tf.reshape(dct_blocks, (*dct_blocks.shape[:-2], bs)), zigzag_order, axis=-1)
        dct_img = tf.transpose(dct_img, perm=(0, 3, 1, 2))
        return dct_img

    def dct_dequantization(dct_img: tf.Tensor, img_shape: tf.TensorShape, quantization_matrix: tf.Tensor) -> tf.Tensor:
        dct_img = tf.transpose(dct_img, perm=(0, 2, 3, 1))
        dct_blocks = tf.reshape(tf.gather(dct_img, reverse_zigzag_order, axis=-1), (*dct_img.shape[:-1], *block_size))
        dct_blocks = dct_blocks * tf.reshape(quantization_matrix, block_size)
        patches = idct_2d(dct_blocks)
        return tf.reshape(tf.transpose(patches, perm=(0, 1, 3, 2, 4)), (-1, *img_shape))

    def encode_sequence(y_dct_img: tf.Tensor, u_dct_img: tf.Tensor, v_dct_img: tf.Tensor) -> tf.Tensor:
        y_dct_seq = dense_to_seq(y_dct_img)
        u_dct_seq = chroma_slope * dense_to_seq(u_dct_img) + u_offset
        v_dct_seq = chroma_slope * dense_to_seq(v_dct_img) + v_offset
        dct_seq = tf.concat((y_dct_seq, u_dct_seq, v_dct_seq), axis=1)
        dense_dct_seq = dct_seq.to_tensor(default_value=-1, shape=(dct_seq.shape[0], max_seq_len, dct_seq.shape[-1]))
        return dense_dct_seq

    def decode_sequence(dense_dct_seq: tf.Tensor, luma_dct_img_shape: tf.TensorShape,
                        chroma_dct_img_shape: tf.TensorShape) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dct_seq = tf.RaggedTensor.from_tensor(dense_dct_seq, padding=[-1, -1, -1, -1])
        y_dct_seq = tf.ragged.boolean_mask(dct_seq, dct_seq[..., 0] < bs)
        u_dct_seq = tf.ragged.boolean_mask(dct_seq, tf.logical_and(dct_seq[..., 0] >= bs, dct_seq[..., 0] < 2 * bs))
        u_dct_seq = (u_dct_seq - u_offset) / chroma_slope
        v_dct_seq = tf.ragged.boolean_mask(dct_seq, dct_seq[..., 2] >= 2 * bs)
        v_dct_seq = (v_dct_seq - v_offset) / chroma_slope
        y_dct_img = seq_to_dense(y_dct_seq, luma_dct_img_shape)
        u_dct_img = seq_to_dense(u_dct_seq, chroma_dct_img_shape)
        v_dct_img = seq_to_dense(v_dct_seq, chroma_dct_img_shape)
        return y_dct_img, u_dct_img, v_dct_img

    def jpeg_encode(image: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.TensorShape, tf.TensorShape, tf.TensorShape, tf.TensorShape]:
        yuv_image = pad(tf.image.rgb_to_yuv(image))
        y, u, v = tf.split(yuv_image, num_or_size_splits=3, axis=-1)
        if chroma_subsample:
            chroma_size = tf.TensorShape([dim // 2 for dim in u.shape[1:-1]])
            u = tf.image.resize(u, size=chroma_size)
            v = tf.image.resize(v, size=chroma_size)
        y_dct_img = dct_quantization(y, t_luma_tf)
        u_dct_img = dct_quantization(u, t_chroma_tf)
        v_dct_img = dct_quantization(v, t_chroma_tf)

        dense_dct_seq = encode_sequence(y_dct_img, u_dct_img, v_dct_img)

        return dense_dct_seq, y.shape[1:], u.shape[1:], y_dct_img.shape[1:], u_dct_img.shape[1:]

    def jpeg_decode(dense_dct_seq: tf.Tensor, luma_shape: tf.TensorShape, chroma_shape: tf.TensorShape,
                    luma_dct_shape: tf.TensorShape, chroma_dct_shape: tf.TensorShape) -> tf.Tensor:
        y_dct_img, u_dct_img, v_dct_img = decode_sequence(dense_dct_seq, luma_dct_shape, chroma_dct_shape)
        y = dct_dequantization(y_dct_img, luma_shape, t_luma_tf)
        u = dct_dequantization(u_dct_img, chroma_shape, t_chroma_tf)
        v = dct_dequantization(v_dct_img, chroma_shape, t_chroma_tf)

        if chroma_subsample:
            chroma_size = tf.TensorShape([dim * 2 for dim in u.shape[1:-1]])
            u = tf.image.resize(u, size=chroma_size)
            v = tf.image.resize(v, size=chroma_size)
        yuv_image = tf.concat((y, u, v), axis=-1)
        return tf.image.yuv_to_rgb(yuv_image)

    return jpeg_encode, jpeg_decode

# image = misc.face()
# plt.imshow(image)
# plt.show()
# encode, decode = get_jpeg_encode_decode_fns(max_seq_len=300, quality=20)
# tf_image = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
# # tf_image = tf.random.uniform((10, 17, 16, 3)) * 255
#
# ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)
#
# l = []
# for i, (image, _) in enumerate(iter(ds_train)):
#     if i == 10:
#         break
#     l.append(image)
#
# tf_image = tf.cast(tf.repeat(tf.convert_to_tensor(l), 3, axis=-1), tf.float32)
# dec = decode(*encode(tf_image))
# plt.imshow(np.clip(dec.numpy()[0], a_min=0, a_max=255).astype(np.int32))
# plt.show()
