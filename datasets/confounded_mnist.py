from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.typing import NDArray
from skimage import draw, morphology, transform

from components.stax_extension import Shape
from datasets.morphomnist import skeleton
from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import ConfoundingFn, get_marginal_datasets, IMAGE, image_gallery, load_cached_dataset
from datasets.utils import get_diagonal_confusion_matrix, get_uniform_confusion_matrix
from datasets.utils import MarginalDistribution
from datasets.utils import Scenario


def function_dict_to_confounding_fn(function_dict: Dict[int, Callable[[IMAGE], IMAGE]],
                                    cm: NDArray[np.float_]) -> ConfoundingFn:
    def apply_fn(image: IMAGE, confounder: int) -> Tuple[IMAGE, int]:
        idx = np.random.choice(cm.shape[1], p=cm[confounder])
        return function_dict[idx](image), idx

    return apply_fn


def get_thickening_fn(amount: float = 1.2) -> Callable[[IMAGE], IMAGE]:
    def apply_fn(image: IMAGE) -> IMAGE:
        morph = ImageMorphology(image[..., 0])
        radius = int(amount * morph.scale * morph.mean_thickness / 2.)
        return np.array(np.expand_dims(morphology.dilation(image[..., 0], morphology.disk(radius)), axis=-1))

    return apply_fn


def get_thinning_fn(amount: float = .7) -> Callable[[IMAGE], IMAGE]:
    def apply_fn(image: IMAGE) -> IMAGE:
        morph = ImageMorphology(image[..., 0])
        radius = int(amount * morph.scale * morph.mean_thickness / 2.)
        return np.array(np.expand_dims(morphology.erosion(image[..., 0], morphology.disk(radius)), axis=-1))

    return apply_fn


def get_swell_fn(strength: float = 3, radius: float = 7) -> Callable[[IMAGE], IMAGE]:
    def _warp(xy: IMAGE, morph: ImageMorphology) -> IMAGE:
        loc_sampler = skeleton.LocationSampler()
        centre = loc_sampler.sample(morph)[::-1]
        _radius = (radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale
        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / _radius) ** (strength - 1)
        weight[distance > _radius] = 1.
        return np.array(centre + weight[:, None] * offset_xy)

    def swell(image: IMAGE) -> IMAGE:
        assert image.ndim == 3 and image.shape[-1] == 1
        morph = ImageMorphology(image[..., 0])
        return np.array(np.expand_dims(transform.warp(image[..., 0], lambda xy: _warp(xy, morph)), axis=-1))

    return swell


def get_fracture_fn(thickness: float = 1.5, prune: float = 2, num_frac: int = 3) -> Callable[[IMAGE], IMAGE]:
    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def _endpoints(morph: ImageMorphology, centre: NDArray) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
        angle = skeleton.get_angle(morph.skeleton, *centre, _ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + _FRAC_EXTENSION * morph.scale
        angle += np.pi / 2.  # Perpendicular to the skeleton
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    def _draw_line(img: IMAGE, p0: NDArray[np.int_], p1: NDArray[np.int_], brush: NDArray[np.bool_]) -> None:
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush

    def fracture(image: IMAGE) -> IMAGE:
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
        return np.array(np.expand_dims(frac_img[r:-r, r:-r], axis=-1))

    return fracture


def get_colourise_fn(cm: NDArray[np.float_]) -> ConfoundingFn:
    colours = tf.constant(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                           (0, 1, 1), (1, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5)))

    def apply_fn(image: IMAGE, confounder: int) -> Tuple[IMAGE, int]:
        idx = np.random.choice(cm.shape[1], p=cm[confounder])
        colour = colours[idx]
        return np.array(np.repeat(image, 3, axis=-1) * np.array(colour)).astype(np.uint8), int(idx)

    return apply_fn


def tf_randint(minval: int, maxval: int, shape: Tuple = ()) -> tf.Tensor:
    return tf.random.uniform(minval=minval, maxval=maxval, dtype=tf.int32, shape=shape)


def random_crop_and_rescale(image: tf.Tensor, fractions: Tuple[float, float] = (.2, .2)) -> tf.Tensor:
    shape = image.shape[:-1]
    start = tuple(tf_randint(minval=0, maxval=int(s * fpd / 2.)) for s, fpd in zip(shape, fractions))
    stop = tuple(tf_randint(minval=int(s * (1. - fpd / 2.)), maxval=s) for s, fpd in zip(shape, fractions))
    slices = tuple((slice(_start, _stop) for _start, _stop in zip(start, stop)))
    cropped_image = image[slices]
    return tf.image.resize(cropped_image, size=shape)


def get_encode_fn(parent_dims: Dict[str, int]) \
        -> Callable[[tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
    def encode_fn(image: tf.Tensor, patents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = (tf.cast(image, tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        patents = {parent: tf.one_hot(value, parent_dims[parent]) for parent, value in patents.items()}
        return image, patents

    return encode_fn


def show_images(dataset: tf.data.Dataset, title: str) -> None:
    data = iter(dataset.batch(128)).__next__()
    order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
    plt.imshow(image_gallery(data[0].numpy()[order], num_images_to_display=128))
    plt.title(title)
    plt.show(block=False)


def create_confounded_mnist_dataset(data_dir: Path,
                                    dataset_name: str,
                                    train_confounding_fns: List[ConfoundingFn],
                                    test_confounding_fns: List[ConfoundingFn],
                                    parent_dims: Dict[str, int],
                                    de_confound: bool) \
        -> Tuple[Dict[FrozenSet[str], tf.data.Dataset], tf.data.Dataset, Dict[str, MarginalDistribution], Shape]:
    input_shape = (-1, 28, 28, 3)
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False,
                                  data_dir=f'{str(data_dir)}/mnist', as_supervised=True)

    encode_fn = get_encode_fn(parent_dims)
    dataset_dir = Path(f'{str(data_dir)}/{dataset_name}')
    train_data, train_parents = load_cached_dataset(dataset_dir / 'train', ds_train, train_confounding_fns, parent_dims)
    train_data = train_data.map(encode_fn)

    test_data, _ = load_cached_dataset(dataset_dir / 'test', ds_test, test_confounding_fns, parent_dims)
    test_data = test_data.map(encode_fn)
    # Get unconfounded datasets by looking at the parents
    train_data_dict, marginals = get_marginal_datasets(train_data, train_parents, parent_dims)
    train_data_dict = train_data_dict if de_confound else dict.fromkeys(train_data_dict.keys(), train_data)

    # for key, dataset in train_data_dict.items():
    #     show_images(dataset, f'train set {str(key)}')
    # show_images(test_data, f'test set')

    train_data_dict = {
        key: dataset.map(lambda image, parents: (random_crop_and_rescale(image, fractions=(.1, .1)), parents))
        for key, dataset in train_data_dict.items()}

    return train_data_dict, test_data, marginals, input_shape


def digit_colour_scenario(data_dir: Path, confound: bool, de_confound: bool) -> Scenario:
    assert not (not confound and de_confound)
    parent_dims = {'digit': 10, 'colour': 10}
    is_invertible = {'digit': False, 'colour': True}
    test_colourise_cm = get_uniform_confusion_matrix(10, 10)
    train_colourise_cm = get_diagonal_confusion_matrix(10, 10, noise=.1) if confound else test_colourise_cm
    train_colourise_fn = get_colourise_fn(train_colourise_cm)
    test_colourise_fn = get_colourise_fn(test_colourise_cm)
    train_confounding_fns = [train_colourise_fn]
    test_confounding_fns = [test_colourise_fn]
    dataset_name = 'mnist_digit_colour' + ('_confounded' if confound else '')
    train_datasets, test_dataset, marginals, input_shape = \
        create_confounded_mnist_dataset(data_dir, dataset_name, train_confounding_fns, test_confounding_fns,
                                        parent_dims, de_confound)
    return train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape


def digit_fracture_colour_scenario(data_dir: Path, confound: bool, de_confound: bool) -> Scenario:
    assert not (not confound and de_confound)
    parent_dims = {'digit': 10, 'fracture': 2, 'colour': 10}
    is_invertible = {'digit': False, 'fracture': False, 'colour': True}

    even_heavy_cm = np.zeros(shape=(10, 2))
    even_heavy_cm[0:-1:2] = (.1, .9)
    even_heavy_cm[1::2] = (.9, .1)

    test_fracture_cm = get_uniform_confusion_matrix(10, 2)
    test_colourise_cm = get_uniform_confusion_matrix(10, 10)
    train_fracture_cm = even_heavy_cm if confound else test_fracture_cm
    train_colourise_cm = get_diagonal_confusion_matrix(10, 10, noise=.1) if confound else test_colourise_cm

    function_dict = {0: lambda x: x, 1: get_fracture_fn(num_frac=1)}
    train_fracture_fn = function_dict_to_confounding_fn(function_dict, train_fracture_cm)
    test_fracture_fn = function_dict_to_confounding_fn(function_dict, test_fracture_cm)
    train_colourise_fn = get_colourise_fn(train_colourise_cm)
    test_colourise_fn = get_colourise_fn(test_colourise_cm)

    train_confounding_fns = [train_fracture_fn, train_colourise_fn]
    test_confounding_fns = [test_fracture_fn, test_colourise_fn]
    dataset_name = 'mnist_digit_fracture_colour' + ('_confounded' if confound else '')
    train_datasets, test_dataset, marginals, input_shape = \
        create_confounded_mnist_dataset(data_dir, dataset_name, train_confounding_fns, test_confounding_fns,
                                        parent_dims, de_confound)
    return train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape
