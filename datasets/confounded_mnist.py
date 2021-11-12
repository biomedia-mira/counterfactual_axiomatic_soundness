from pathlib import Path
from typing import Callable, Dict, FrozenSet, Tuple
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from skimage import draw, morphology, transform
from tqdm import tqdm

from components.stax_extension import Shape
from datasets.morphomnist import skeleton
from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import get_diagonal_confusion_matrix, get_uniform_confusion_matrix
from datasets.utils import get_marginal_datasets
from datasets.utils import image_gallery

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


def get_fracture_fn(thickness: float = 1.5, prune: float = 2, num_frac: int = 3) -> Callable[[np.ndarray], np.ndarray]:
    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def _endpoints(morph, centre):
        angle = skeleton.get_angle(morph.skeleton, *centre, _ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + _FRAC_EXTENSION * morph.scale
        angle += np.pi / 2.  # Perpendicular to the skeleton
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush

    def fracture(image: np.ndarray) -> np.ndarray:
        morph = ImageMorphology(image[..., 0])
        loc_sampler = skeleton.LocationSampler(prune, prune)

        up_thickness = thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        frac_img = np.pad(image[..., 0], pad_width=r, mode='constant', constant_values=False)
        try:
            centres = loc_sampler.sample(morph, num_frac)
        except ValueError:  # Skeleton vanished with pruning, attempt without
            centres = skeleton.LocationSampler().sample(morph, num_frac)
        for centre in centres:
            p0, p1 = _endpoints(morph, centre)
            _draw_line(frac_img, p0, p1, brush)
        return np.expand_dims(frac_img[r:-r, r:-r], axis=-1)

    return fracture


def get_swell_fn(strength: float = 3, radius: float = 7) -> Callable[[np.ndarray], np.ndarray]:
    def warp(xy: np.ndarray, morph: ImageMorphology, strength, radius) -> np.ndarray:
        loc_sampler = skeleton.LocationSampler()
        centre = loc_sampler.sample(morph)[::-1]
        radius = (radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale
        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy

    def swell(image: np.ndarray) -> np.ndarray:
        assert image.ndim == 3 and image.shape[-1] == 1
        morph = ImageMorphology(image[..., 0])
        return np.expand_dims(transform.warp(image[..., 0], lambda xy: warp(xy, morph, strength, radius)), axis=-1)

    return swell


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
        # _image = (tf.image.resize(tf.cast(_image, tf.float32), (32, 32)) - tf.constant(127.5)) / tf.constant(127.5)
        _image = (tf.cast(_image, tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        _parents = {parent: tf.one_hot(value, parent_dims[parent]) for parent, value in _parents.items()}
        return _image, _parents

    dataset = dataset.map(encode)
    return dataset, parents


def create_confounded_mnist_dataset() -> Tuple[
    Dict[FrozenSet[str], tf.data.Dataset], tf.data.Dataset, Dict[str, int], Dict[str, np.ndarray], Shape]:
    input_shape = (-1, 28, 28, 3)
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, as_supervised=True)
    train_mechanisms, test_mechanisms, parent_dims = get_apply_mechanisms_fn()

    dataset_dir = Path('./data/confounded_mnist')
    train_data, train_parents = get_dataset(dataset_dir / 'train', ds_train, train_mechanisms, parent_dims)
    test_data, _ = get_dataset(dataset_dir / 'test', ds_test, test_mechanisms, parent_dims)
    test_data = test_data.shuffle(buffer_size=1000)
    # Get unconfounded datasets by looking at the parents
    train_data, marginals = get_marginal_datasets(train_data, train_parents, parent_dims)

    for key, dataset in train_data.items():
        data = iter(dataset.batch(128)).__next__()
        order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
        plt.imshow(image_gallery(127.5 * data[0].numpy()[order] + 127.5, num_images_to_display=128))
        plt.title(f'train set {str(key)}')
        plt.show()

    data = iter(test_data.batch(128)).__next__()
    order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
    plt.imshow(image_gallery(127.5 * data[0].numpy()[order] + 127.5, num_images_to_display=128))
    plt.title(f'test set')
    plt.show()

    return train_data, test_data, parent_dims, marginals, input_shape
