from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Tuple, Union

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

tf.config.experimental.set_visible_devices([], 'GPU')


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


def get_colorize_fn(cm: NDArray[np.float_]) -> ConfoundingFn:
    colors = tf.constant(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                          (0, 1, 1), (1, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5)))

    def apply_fn(image: IMAGE, confounder: int) -> Tuple[IMAGE, int]:
        idx = np.random.choice(cm.shape[1], p=cm[confounder])
        color = colors[idx]
        return np.array(np.repeat(image, 3, axis=-1) * np.array(color)).astype(np.uint8), int(idx)

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


def decode_fn(image: Union[tf.Tensor, NDArray]) -> Union[tf.Tensor, NDArray]:
    return 127.5 * image + 127.5


def show_images(dataset: tf.data.Dataset, title: str) -> None:
    data = iter(dataset.batch(128)).__next__()
    order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
    plt.imshow(image_gallery(decode_fn(data[0].numpy()[order]), num_images_to_display=128))
    plt.title(title)
    plt.show()


def create_confounded_mnist_dataset(data_dir: str,
                                    train_confounding_fns: List[ConfoundingFn],
                                    test_confounding_fns: List[ConfoundingFn],
                                    parent_dims: Dict[str, int]) \
        -> Tuple[Dict[FrozenSet[str], tf.data.Dataset], tf.data.Dataset, Dict[str, NDArray], Shape]:
    input_shape = (-1, 28, 28, 3)
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, data_dir=f'{data_dir}/mnist',
                                  as_supervised=True)

    encode_fn = get_encode_fn(parent_dims)
    dataset_dir = Path(f'{data_dir}/confounded_mnist')
    train_data, train_parents = load_cached_dataset(dataset_dir / 'train', ds_train, train_confounding_fns, parent_dims)
    train_data = train_data.map(encode_fn)

    test_data, _ = load_cached_dataset(dataset_dir / 'test', ds_test, test_confounding_fns, parent_dims)
    test_data = test_data.map(encode_fn).shuffle(buffer_size=1000)
    # Get unconfounded datasets by looking at the parents
    train_data, marginals = get_marginal_datasets(train_data, train_parents, parent_dims)

    for key, dataset in train_data.items():
        show_images(dataset, f'train set {str(key)}')
    show_images(test_data, f'test set')

    train_data = {
        key: dataset.map(lambda image, parents: (random_crop_and_rescale(image, fractions=(.3, .3)), parents))
        for key, dataset in train_data.items()}

    return train_data, test_data, marginals, input_shape
