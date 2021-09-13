import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from scipy.ndimage import gaussian_filter

from datasets.jpeg import get_jpeg_encode_decode_fns
from datasets.morphomnist.perturb import ImageMorphology, Swelling
from datasets.utils import get_unconfounded_datasets, image_gallery
from pathlib import Path

tf.config.experimental.set_visible_devices([], 'GPU')

MechanismFn = Callable[[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]]
##
from skimage import transform

from datasets.morphomnist import skeleton
from datasets.morphomnist.morpho import ImageMorphology


def warp(xy: np.ndarray, morph: ImageMorphology, strength, radius) -> np.ndarray:
    loc_sampler = skeleton.LocationSampler()
    centre = loc_sampler.sample(morph)[::-1]
    radius = (radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale
    offset_xy = xy - centre
    distance = np.hypot(*offset_xy.T)
    weight = (distance / radius) ** (strength - 1)
    weight[distance > radius] = 1.
    return centre + weight[:, None] * offset_xy


def ___swell(image: np.ndarray, strength: float = 3, radius: float = 7):
    assert image.ndim == 3 and image.shape[-1] == 1
    morph = ImageMorphology(image[..., 0])
    return np.expand_dims(transform.warp(image[..., 0], lambda xy: warp(xy, morph, strength, radius)), axis=-1)


##

def get_jpeg_encoding_decoding_fns(max_seq_len: int, image_shape: Tuple[int, int, int]):
    encode_fn, decode_fn = get_jpeg_encode_decode_fns(max_seq_len=max_seq_len, block_size=(8, 8), quality=50,
                                                      chroma_subsample=False)
    dummy = tf.convert_to_tensor(np.expand_dims(np.zeros(image_shape, dtype=np.float32), axis=0))
    seq, luma_shape, chroma_shape, luma_dct_shape, chroma_dct_shape = encode_fn(dummy)

    norm_factor = (luma_dct_shape[0] * 3 - 1, luma_dct_shape[1], luma_dct_shape[2], 255)
    tf_norm_factor = tf.broadcast_to(tf.convert_to_tensor(norm_factor, dtype=tf.float32) - 1., seq.shape)

    def jpeg_decode_fn(dense_dct_seq: np.ndarray) -> np.ndarray:
        return decode_fn(tf.convert_to_tensor(dense_dct_seq * tf_norm_factor), luma_shape, chroma_shape, luma_dct_shape,
                         chroma_dct_shape).numpy()

    def jpeg_encode_fn(image: tf.Tensor) -> tf.Tensor:
        return encode_fn(image)[0] / tf_norm_factor

    return jpeg_encode_fn, jpeg_decode_fn


def rgb_decode_fn(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255., a_min=0, a_max=255).astype(dtype=np.int32)


def rgb_encode_fn(image: tf.Tensor) -> tf.Tensor:
    return image / tf.constant(255.)


def get_uniform_confusion_matrix(num_rows: int, num_columns: int) -> np.ndarray:
    return np.ones((num_rows, num_columns)) / num_columns


def get_random_confusion_matrix(num_rows: int, num_columns: int, temperature: float = .1, seed: int = 1) -> np.ndarray:
    random_state = np.random.RandomState(seed=seed)
    logits = random_state.random(size=(num_rows, num_columns))
    tmp = np.exp(logits / temperature)
    return tmp / tmp.sum(1, keepdims=True)


def get_diagonal_confusion_matrix(num_rows: int, num_columns: int, noise: float = 0.) -> np.ndarray:
    assert num_rows == num_columns
    return (np.eye(num_rows) * (1. - noise)) + (np.ones((num_rows, num_rows)) - np.eye(num_rows)) * noise / (
            num_rows - 1)


def get_colorize_fn(digit: np.ndarray, cm: np.ndarray):
    indices = tf.constant(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), digit)), dtype=tf.int64)
    colors = tf.constant(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                          (0, 1, 1), (1, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5)))

    def apply_fn(index: tf.Tensor, image: tf.Tensor, parents: Dict[str, tf.Tensor]) \
            -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        idx = indices[index]
        color = colors[idx]
        image = tf.concat([image] * 3, axis=-1) * tf.reshape(color, (1, 1, 3))
        return index, image, {**parents, 'color': idx}

    return apply_fn, {'color': indices.numpy()}


def get_swelling_fn(digit: np.ndarray, cm: np.ndarray):
    indices = tf.constant(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), digit)), dtype=tf.int64)

    def warp(xy: np.ndarray, morph: ImageMorphology, strength, radius) -> np.ndarray:
        loc_sampler = skeleton.LocationSampler()
        centre = loc_sampler.sample(morph)[::-1]
        radius = (radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale
        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy

    def swell(image: np.ndarray, strength: float = 3, radius: float = 7):
        assert image.ndim == 3 and image.shape[-1] == 1
        morph = ImageMorphology(image[..., 0])
        return np.expand_dims(transform.warp(image[..., 0], lambda xy: warp(xy, morph, strength, radius)), axis=-1)

    def apply_fn(index: tf.Tensor, image: tf.Tensor, parents: Dict[str, tf.Tensor]) \
            -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        idx = indices[index]
        image = tf.cond(idx == 0, lambda: image, lambda: tf.numpy_function(swell, [image], tf.float32))
        return index, image, {**parents, 'swell': idx}

    return apply_fn, {'swell': indices.numpy()}


def get_mechanisms(digit: np.ndarray) -> Tuple[List[MechanismFn], Dict[str, np.ndarray]]:
    # even digits have much higher chance of swelling
    train_swell_cm = np.zeros(shape=(10, 2))
    train_swell_cm[0:-1:2] = (.1, .9)
    train_swell_cm[1::2] = (.9, .1)
    swelling_fn, swell = get_swelling_fn(digit, train_swell_cm)

    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1)
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    colorize_fn, color = get_colorize_fn(digit, train_colorize_cm)

    parents_np = {'digit': digit, **swell, **color}
    return [swelling_fn, colorize_fn], parents_np


def apply_mechanisms_to_dataset(dataset: tf.data.Dataset, mechanisms: List[MechanismFn]) -> tf.data.Dataset:
    dataset = dataset.map(lambda image, label: (tf.cast(image, tf.float32), {'digit': label}))
    dataset = dataset.enumerate().map(lambda index, data: (index, data[0], data[1]))
    for mechanism in mechanisms:
        dataset = dataset.map(mechanism, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def create_confounded_mnist_dataset(batch_size: int, debug: bool = True, jpeg_encode: bool = True):
    cache_dir = Path('./data/confounded_mnist')
    parent_dims = {'digit': 10, 'swell': 2, 'color': 10}
    image_shape = (28, 28, 3)
    seq_len = 350
    seq_shape = (seq_len, 4)

    if not os.path.exists(cache_dir):
        ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised=True)
        train_targets = np.array([y for x, y in ds_train.as_numpy_iterator()])
        test_targets = np.array([y for x, y in ds_train.as_numpy_iterator()])
        # Confound the dataset artificially
        mechanisms, parents_np = get_mechanisms(train_targets)
        dataset = apply_mechanisms_to_dataset(ds_train, mechanisms)
        tf.data.experimental.save(dataset, str(cache_dir), compression='GZIP')
        np.save(str(cache_dir / 'parents.npy'), parents_np)
    else:
        dataset = tf.data.experimental.load(str(cache_dir), compression='GZIP')
        parents_np = np.load(str(cache_dir / 'parents.npy'), allow_pickle=True).item()

    if jpeg_encode:
        raise NotImplementedError
        # img_encode_fn, img_decode_fn = get_jpeg_encoding_decoding_fns(seq_len, image_shape)
        # input_shape = (-1, *seq_shape)
    else:
        img_encode_fn, img_decode_fn, input_shape = rgb_encode_fn, rgb_decode_fn, (-1, *image_shape)

    # Get unconfounded datasets by looking at the parents
    dataset, marginals = get_unconfounded_datasets(dataset, parents_np, parent_dims, batch_size, img_encode_fn)

    if debug:
        flag = False
        for el in dataset:
            if not flag:
                for key, data in el.items():
                    order = np.argsort(np.argmax(data[1]['digit'], axis=1))
                    plt.imshow(image_gallery(img_decode_fn(data[0])[order], num_images_to_display=256))
                    plt.title(str(key))
                    plt.show()
                flag = True
    return dataset, parent_dims, marginals, img_decode_fn, input_shape
