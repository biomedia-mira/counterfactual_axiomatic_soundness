from functools import partial
from datasets.jpeg import get_jpeg_encode_decode_fns
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

image = misc.face()
plt.imshow(image)
plt.show()
encode, decode = get_jpeg_encode_decode_fns(max_seq_len=15000, quality=20)
tf_image = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.float32)
# tf_image = tf.random.uniform((10, 17, 16, 3)) * 255

# ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)
# l = []
# for i, (image, _) in enumerate(iter(ds_train)):
#     if i == 10:
#         break
#     l.append(image)
# tf_image = tf.cast(tf.repeat(tf.convert_to_tensor(l), 3, axis=-1), tf.float32)

dense_dct_seq, luma_shape, chroma_shape, luma_dct_shape, chroma_dct_shape = encode(tf_image)
# rep = tf.concat([dense_dct_seq[:, :5000]] * 3, axis=1)
dec = decode(dense_dct_seq, luma_shape, chroma_shape, luma_dct_shape, chroma_dct_shape)


plt.imshow(np.clip(dec.numpy()[0], a_min=0, a_max=255).astype(np.int32))
plt.show()

# def lexsort_based(seq):
#     order = jnp.lexsort(seq[..., :-1].T)
#     sorted_indices, sorted_values = seq[order, :-1], seq[order, -1]
#
#     diff = jnp.any(jnp.diff(sorted_indices, axis=0), 1)
#
#     negative_mask = jnp.any(sorted_indices == -1, 1)
#     unique_mask = jnp.append(True, diff)
#
#     segment_ids = jnp.cumsum(jnp.append(False, diff)) - 1
#     segment_ids = jax.ops.index_update(segment_ids, negative_mask, -1)
#     new_values = jax.ops.segment_sum(sorted_values, segment_ids, num_segments=len(seq))
#
#
#
#     new_indices = jax.ops.segment_sum(sorted_indices, jnp.cumsum(unique_mask), num_segments=len(seq))
#     new_indices = jnp.ones_like(sorted_indices) * -1
#     jax.ops.index_update(new_indices, jnp.arange(len(seq)), sorted_indices[unique_mask])
#
#     return sorted_data[row_mask]


# tools to make valid jpeg sequences (in jax)
# def average_duplicates(seq):
#     folded, indices, counts = jnp.unique(seq[..., :-1], return_inverse=True, return_counts=True, axis=0)
#     output = jnp.zeros(folded.shape[0])
#     output = jax.ops.index_add(output, indices, seq[..., -1])
#     output = output / counts
#     return jnp.concatenate((folded, output[..., jnp.newaxis]), axis=-1)