from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.typing import NDArray

from components.stax_extension import Shape
from datasets.augmentation import random_crop_and_rescale
from datasets.confounding import get_diagonal_confusion_matrix, get_marginal_datasets, get_uniform_confusion_matrix, \
    Mechanism
from datasets.mnist_mechanisms import function_dict_to_mechanism, get_colorize_fn, get_thickening_fn, get_thinning_fn
from datasets.utils import image_gallery, load_cached_dataset


def get_encode_fn(parent_dims: Dict[str, int]) \
        -> Callable[[tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
    def encode_fn(image: tf.Tensor, patents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = (tf.cast(image, tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        patents = {parent: tf.one_hot(value, parent_dims[parent]) for parent, value in patents.items()}
        return image, patents

    return encode_fn


def decode_fn(image: Union[tf.Tensor, NDArray]) -> Union[tf.Tensor, NDArray]:
    return 127.5 * image + 127.5


def show_images(dataset: tf.data.Dataset, title: str) -> None:
    data = iter(dataset.batch(128)).__next__()
    order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
    plt.imshow(image_gallery(decode_fn(data[0].numpy()[order]), num_images_to_display=128))
    plt.title(title)
    plt.show()


def get_apply_mechanisms_fn() -> Tuple[List[Mechanism], List[Mechanism], Dict[str, int]]:
    parent_dims = {'digit': 10, 'thickness': 2, 'color': 10}

    # even digits have much higher chance of swelling
    train_thickening_cm = np.zeros(shape=(10, 2))
    train_thickening_cm[0:-1:2] = (.1, .9)
    train_thickening_cm[1::2] = (.9, .1)
    function_dict = {0: get_thinning_fn(), 1: get_thickening_fn()}
    train_thickening_fn = function_dict_to_mechanism(function_dict, train_thickening_cm)

    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1)
    train_colorize_fn = get_colorize_fn(train_colorize_cm)

    train_mechanisms = [train_thickening_fn, train_colorize_fn]

    test_thickening_cm = get_uniform_confusion_matrix(10, 2)
    test_thickening_fn = function_dict_to_mechanism(function_dict, test_thickening_cm)
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    test_colorize_fn = get_colorize_fn(test_colorize_cm)
    test_mechanisms = [test_thickening_fn, test_colorize_fn]

    return train_mechanisms, test_mechanisms, parent_dims


def create_confounded_mnist_dataset(data_dir: str = './data') \
        -> Tuple[Dict[FrozenSet[str], tf.data.Dataset], tf.data.Dataset, Dict[str, int], Dict[str, NDArray], Shape]:
    tf.config.experimental.set_visible_devices([], 'GPU')
    input_shape = (-1, 28, 28, 3)
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, data_dir=f'{data_dir}/mnist')
    train_mechanisms, test_mechanisms, parent_dims = get_apply_mechanisms_fn()

    encode_fn = get_encode_fn(parent_dims)
    dataset_dir = Path(f'{data_dir}/confounded_mnist')
    train_data, train_parents = load_cached_dataset(dataset_dir / 'train', ds_train, train_mechanisms, parent_dims)
    train_data = train_data.map(encode_fn)

    test_data, _ = load_cached_dataset(dataset_dir / 'test', ds_test, test_mechanisms, parent_dims)
    test_data = test_data.map(encode_fn).shuffle(buffer_size=1000)
    # Get unconfounded datasets by looking at the parents
    train_data, marginals = get_marginal_datasets(train_data, train_parents, parent_dims)

    for key, dataset in train_data.items():
        show_images(dataset, f'train set {str(key)}')
    show_images(test_data, f'test set')

    train_data = {
        key: dataset.map(lambda image, parents: (random_crop_and_rescale(image, fractions=(.3, .3)), parents))
        for key, dataset in train_data.items()}

    return train_data, test_data, parent_dims, marginals, input_shape
