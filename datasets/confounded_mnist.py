from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from more_itertools import powerset
from skimage import morphology
from tqdm import tqdm

from components.typing import Shape
from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import get_diagonal_confusion_matrix, get_uniform_confusion_matrix
from datasets.utils import get_marginal_datasets, image_gallery

tf.config.experimental.set_visible_devices([], 'GPU')

Mechanism = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def function_dict_to_apply_fn(function_dict: Dict[int, Callable[[np.ndarray], np.ndarray]], cm: np.array) -> Mechanism:
    def apply_fn(image: np.ndarray, digit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(cm.shape[1], p=cm[digit])
        return function_dict[idx](image), idx

    return apply_fn


def get_thickening_fn(amount: float = 1.2) -> Callable[[np.ndarray], np.ndarray]:
    def thicken(image: np.ndarray) -> np.ndarray:
        morph = ImageMorphology(image[..., 0])
        radius = int(amount * morph.scale * morph.mean_thickness / 2.)
        return np.expand_dims(morphology.dilation(image[..., 0], morphology.disk(radius)), axis=-1)

    return thicken


def get_thinning_fn(amount: float = .7) -> Callable[[np.ndarray], np.ndarray]:
    def thin(image: np.ndarray) -> np.ndarray:
        morph = ImageMorphology(image[..., 0])
        radius = int(amount * morph.scale * morph.mean_thickness / 2.)
        return np.expand_dims(morphology.erosion(image[..., 0], morphology.disk(radius)), axis=-1)

    return thin


def get_colorize_fn(cm: np.ndarray) -> Mechanism:
    colors = tf.constant(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                          (0, 1, 1), (1, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5)))

    def apply_fn(image: np.ndarray, digit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(cm.shape[1], p=cm[digit])
        color = colors[idx]
        return (np.repeat(image, 3, axis=-1) * np.array(color)).astype(np.uint8), idx

    return apply_fn


def get_apply_mechanisms_fn() -> Tuple[List[Mechanism], List[Mechanism], Dict[str, int]]:
    parent_dims = {'digit': 10, 'thickness': 2, 'color': 10}

    # even digits have much higher chance of swelling
    train_thickening_cm = np.zeros(shape=(10, 2))
    train_thickening_cm[0:-1:2] = (.1, .9)
    train_thickening_cm[1::2] = (.9, .1)
    function_dict = {0: get_thinning_fn(), 1: get_thickening_fn()}
    train_thickening_fn = function_dict_to_apply_fn(function_dict, train_thickening_cm)

    train_colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1)
    train_colorize_fn = get_colorize_fn(train_colorize_cm)

    train_mechanisms = [train_thickening_fn, train_colorize_fn]

    test_thickening_cm = get_uniform_confusion_matrix(10, 2)
    test_thickening_fn = function_dict_to_apply_fn(function_dict, test_thickening_cm)
    test_colorize_cm = get_uniform_confusion_matrix(10, 10)
    test_colorize_fn = get_colorize_fn(test_colorize_cm)
    test_mechanisms = [test_thickening_fn, test_colorize_fn]

    return train_mechanisms, test_mechanisms, parent_dims


def apply_mechanisms_to_dataset(dataset: tf.data.Dataset, mechanisms: List[Mechanism], parent_dims: Dict[str, int]) \
        -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    def apply_fn(__image: np.ndarray, __digit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        __parents = [int(__digit)]
        for mechanism in mechanisms:
            __image, new_p = mechanism(__image, __digit)
            __parents.append(new_p)
        return __image, np.array(__parents).astype(np.int64)

    image_list, parents_list = [], []
    for _image, _digit in tqdm(dataset.as_numpy_iterator()):
        _image, _parents = apply_fn(_image, _digit)
        image_list.append(_image)
        parents_list.append(_parents)
    image = np.array(image_list)
    parents = {key: np.array(parents_list)[:, i] for i, key in enumerate(parent_dims.keys())}
    return image, parents


def get_dataset(dataset_dir: Path, dataset: tf.data.Dataset, mechanisms: List[Mechanism], parent_dims: Dict[str, int]) \
        -> Tuple[tf.data.Dataset, Dict[str, np.ndarray]]:
    parents_path = str(dataset_dir / 'parents.npy')
    images_path = str(dataset_dir / 'images.npy')
    try:
        images = np.load(images_path)
        parents = np.load(parents_path, allow_pickle=True).item()
    except FileNotFoundError:
        images, parents = apply_mechanisms_to_dataset(dataset, mechanisms, parent_dims)
        dataset_dir.mkdir(exist_ok=True, parents=True)
        np.save(images_path, images)
        np.save(parents_path, parents)
    dataset = tf.data.Dataset.from_tensor_slices((images, parents))

    def encode(_image: tf.Tensor, _parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        _image = tf.cast(_image, tf.float32) / tf.constant(255.)
        _parents = {parent: tf.one_hot(value, parent_dims[parent]) for parent, value in _parents.items()}
        return _image, _parents

    dataset = dataset.map(encode)
    return dataset, parents


def setup_dataset_for_training(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return tfds.as_numpy(dataset)


def create_confounded_mnist_dataset(batch_size: int, debug: bool = True) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int], Dict[str, np.ndarray], Shape]:
    input_shape = (-1, 28, 28, 3)
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised=True)
    train_mechanisms, test_mechanisms, parent_dims = get_apply_mechanisms_fn()

    dataset_dir = Path('./data/confounded_mnist')
    train_data, train_parents = get_dataset(dataset_dir / 'train', ds_train, train_mechanisms, parent_dims)
    test_data, _ = get_dataset(dataset_dir / 'test', ds_test, test_mechanisms, parent_dims)

    # Get unconfounded datasets by looking at the parents
    train_data, marginals = get_marginal_datasets(train_data, train_parents, parent_dims)

    test_data = tf.data.Dataset.zip(
        {frozenset(parent_set): test_data for parent_set in powerset(parent_dims.keys())})

    train_data = tfds.as_numpy(train_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    test_data = tfds.as_numpy(test_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
    if debug:
        for el in test_data:
            for key, data in el.items():
                order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
                plt.imshow(image_gallery(255. * data[0][order], num_images_to_display=128))
                plt.title(str(key))
                plt.show()
            break
    return train_data, test_data, parent_dims, marginals, input_shape
