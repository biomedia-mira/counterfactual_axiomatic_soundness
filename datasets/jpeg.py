from typing import Tuple, List

import numpy as np
import tensorflow as tf
import itertools
from functools import partial

from scipy import misc
import imageio
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def zigzag(rows: int, columns: int) -> Tuple[Tuple[int, int], ...]:
    order: List[List[Tuple[int, int]]] = [[] for i in range(rows + columns - 1)]
    for i in range(columns):
        for j in range(rows):
            sum_ = i + j
            if sum_ % 2 == 0:
                order[sum_].insert(0, (i, j))  # add at beginning
            else:
                order[sum_].append((i, j))  # add at end of the list
    return tuple(itertools.chain(*order))


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
            tf.SparseTensor(dense_shape=img_shape, indices=tf.cast(x[..., :-1], tf.int64), values=x[..., 0]))

    return tf.map_fn(_seq_to_dense, seq, fn_output_signature=tf.TensorSpec(shape=img_shape))
    #
    # values = tf.concat([d[..., -1] for d in seq], axis=0)
    # indices = tf.concat([tf.concat((tf.broadcast_to([float(i)], (d.shape[0], 1)), d[..., :-1]), axis=-1) for i, d in
    #                      enumerate(seq)], axis=0)
    # sparse = tf.SparseTensor(dense_shape=img_shape, values=values, indices=tf.cast(indices, tf.int64))
    # sparse = tf.sparse.reorder(sparse)
    # return tf.sparse.to_dense(sparse)


def create_jpeg_encoding_fn(block_size: Tuple[int, int] = (8, 8), quality: float = 50.):
    assert block_size == (8, 8)
    zigzag_order = zigzag(*block_size)
    t_luma_tf = tf.reshape(tf.convert_to_tensor(get_t_luma(quality), dtype=tf.float32), (1, 1, 1, 8, 8))
    t_chroma_tf = tf.reshape(tf.convert_to_tensor(t_chroma, dtype=tf.float32), (1, 1, 1, 8, 8))

    def dct_quantization(component: tf.Tensor, quantization_matrix: tf.Tensor) -> tf.Tensor:
        sizes = (1, *block_size, 1)
        patches = tf.image.extract_patches(component, sizes=sizes, strides=sizes, rates=(1, 1, 1, 1), padding='SAME')
        patches = tf.reshape(patches, shape=(*patches.shape[:3], *block_size))
        dct_blocks = dct_2d(patches)
        dct_blocks = tf.round(dct_blocks / quantization_matrix)
        dct_img = tf.map_fn(lambda x: tf.gather_nd(x, zigzag_order), elems=tf.reshape(dct_blocks, (-1, *block_size)))
        dct_img = tf.reshape(dct_img, (*dct_blocks.shape[:3], block_size[0] * block_size[1]))
        return dct_img

    def adjust_chroma_seq(dct_seq, dct_band: int) -> tf.Tensor:
        assert dct_band in [1, 2]
        slope = tf.convert_to_tensor([2, 2, 1, 1], dtype=tf.float32)
        offset = tf.convert_to_tensor([0, 0, dct_band * block_size[0] * block_size[1], 1], dtype=tf.float32)
        return dct_seq * slope + offset

    def jpeg_encode(image: tf.Tensor) -> Tuple[List[tf.Tensor], tf.TensorShape]:
        image = tf.image.rgb_to_yuv(image)
        y, u, v = tf.split(image, num_or_size_splits=3, axis=-1)
        chroma_size = tf.TensorShape([dim // 2 for dim in u.shape[1:-1]])
        u = tf.image.resize(u, size=chroma_size)
        v = tf.image.resize(v, size=chroma_size)
        y_dct_img = dct_quantization(y, t_luma_tf)
        u_dct_img = dct_quantization(u, t_chroma_tf)
        v_dct_img = dct_quantization(v, t_chroma_tf)

        y_dct_seq = dense_to_seq(y_dct_img)
        u_dct_seq = adjust_chroma_seq(dense_to_seq(u_dct_img), dct_band=1)
        v_dct_seq = adjust_chroma_seq(dense_to_seq(v_dct_img), dct_band=2)
        seq = tf.concat((y_dct_seq, u_dct_seq, v_dct_seq), axis=1)
        dense_seq = seq.to_tensor(default_value=-1)

        seq_2 = tf.RaggedTensor.from_tensor(dense_seq, padding=[-1, -1, -1, -1])
        bs = block_size[0] * block_size[1]
        y_dct_seq_2 = tf.ragged.boolean_mask(seq_2, seq_2[..., 2] < bs)
        u_dct_seq_2 = tf.ragged.boolean_mask(seq_2, tf.logical_and(seq_2[..., 2] >= bs, seq_2[..., 2] < 2 * bs))
        v_dct_seq_2 = tf.ragged.boolean_mask(seq_2, seq_2[..., 2] >= 2 * bs)
        seq_to_dense(y_dct_seq_2, y_dct_img.shape[1:])
        return dense_seq, 0

    def jpeg_decode(dense_seq, img_shape):
        #
        # s[tf.logical_and(s[:, 3] > 64, s[:, 3] < 128)]
        # s[tf.logical_and(s[:, 3] > 64, s[:, 3] < 128)]
        # s[tf.logical_and(s[:, 3] > 64, s[:, 3] < 128)]

        dct_img = seq_to_dense(seq, img_shape)
        h_dct_img, u_dct_img, v_dct_img = tf.split(dct_img, num_or_size_splits=3, axis=-1)

        image = idct_2d(dct_img)
        return tf.reshape(tf.transpose(image, perm=(0, 1, 3, 2, 4)), (10, 16, 16, 1))

    return jpeg_encode, jpeg_decode


image = misc.face()
plt.imshow(image)
plt.show()
encode, decode = create_jpeg_encoding_fn()
tf_image = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
tf_image = tf.random.uniform((10, 28, 28, 3)) * 255

# ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)
#
# l = []
# for i, (image, _) in enumerate(iter(ds_train)):
#     if i == 10:
#         break
#     l.append(image)
#
# tf_image = tf.cast(tf.repeat(tf.convert_to_tensor(l), 3, axis=-1), tf.float32)

seq, img_shape = encode(tf_image)
tf_decoded = decode(seq, img_shape)
