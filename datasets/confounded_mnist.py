import pickle
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Tuple

import jax
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.typing import NDArray
from numpyro.distributions import Normal, TransformedDistribution
from numpyro.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from skimage import morphology, transform
from tensorflow.keras import layers
from tqdm import tqdm

from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import get_simulated_intervention_datasets, IMAGE, image_gallery, ParentDist


def set_colour(image: IMAGE, colour: NDArray) -> IMAGE:
    return np.array(np.repeat(image, 3, axis=-1) * colour).astype(np.uint8)


@lru_cache(maxsize=None)
def _get_disk(radius: int, scale: int):
    mag_radius = scale * radius
    mag_disk = morphology.disk(mag_radius, dtype=np.float64)
    disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1, multichannel=False)
    return disk


def set_thickness(image: IMAGE, target_thickness: float) -> IMAGE:
    morph = ImageMorphology(image[..., 0], scale=16)
    delta = target_thickness - morph.mean_thickness
    radius = int(morph.scale * abs(delta) / 2.)
    disk = _get_disk(radius, scale=16)
    img = morph.binary_image
    if delta >= 0:
        up_scale_image = morphology.dilation(img, disk)
    else:
        up_scale_image = morphology.erosion(img, disk)
    image = morph.downscale(np.float32(up_scale_image))
    return image[..., np.newaxis]


def set_intensity(image: IMAGE, intensity: float) -> IMAGE:
    threshold = 0.5
    img_min, img_max = np.min(image), np.max(image)
    mask = (image >= img_min + (img_max - img_min) * threshold)
    avg_intensity = np.median(image[mask])
    factor = intensity / avg_intensity
    return np.clip(image * factor, 0, 255).astype(np.uint8)


# Thickness intensity model
def thickness_intensity_model(confound: bool, n_samples=None, scale=0.5, invert=False):
    with numpyro.plate('observations', n_samples):
        k1, k2, k3 = random.split(random.PRNGKey(1), 3)
        thickness_transform = ComposeTransform(
            [AffineTransform(-1., 1.), SigmoidTransform(), AffineTransform(1.5, 4.5)])
        thickness_dist = TransformedDistribution(Normal(0., 1.), thickness_transform)
        thickness = numpyro.sample('thickness', thickness_dist, rng_key=k1)
        multiplier = -1 if invert else 1
        # if not confound intensity does not depend on thickness
        _thickness = thickness if confound else numpyro.sample('thickness', thickness_dist, rng_key=k2)
        loc = (_thickness - 2.5) * 2 * multiplier if confound else 0.
        transforms = ComposeTransform([SigmoidTransform(), AffineTransform(64, 191)])
        intensity = numpyro.sample('intensity', TransformedDistribution(Normal(loc, scale), transforms), rng_key=k3)
    return np.array(thickness), np.array(intensity)


def digit_thickness_intensity(ds_train: tf.data.Dataset, ds_test: tf.data.Dataset, confound: bool):
    parent_dists = {'digit': ParentDist(dim=10, is_discrete=True, is_invertible=False),
                    'thickness': ParentDist(dim=1, is_discrete=False, is_invertible=True),
                    'intensity': ParentDist(dim=1, is_discrete=False, is_invertible=True)}
    input_shape = (-1, 28, 28, 1)

    def confound_dataset(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[NDArray, Dict[str, NDArray]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        thickness, intensity = thickness_intensity_model(confound=confound, n_samples=len(dataset))
        images = np.array([set_intensity(set_thickness(image, t), i)
                           for (image, _), t, i in tqdm(zip(dataset.as_numpy_iterator(), thickness, intensity))])
        parents = {'digit': digit, 'thickness': thickness, 'intensity': intensity}
        return images, parents

    train_images, train_parents = confound_dataset(ds_train, confound=confound)
    test_images, test_parents = confound_dataset(ds_test, confound=False)
    return train_images, train_parents, test_images, test_parents, parent_dists, input_shape


# Confounded MNIST
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


def confounded_mnist(data_dir: Path, dataset_name: str, confound: bool, plot: bool = False):
    fn = {'digit_thickness_intensity': digit_thickness_intensity}[dataset_name]
    dataset_path = Path(f'{str(data_dir)}/{dataset_name}' + (f'_confounded' if confound else ''))
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print('Dataset not found, creating new copy...')
        ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False,
                                      data_dir=f'{str(data_dir)}/mnist', as_supervised=True)
        dataset = fn(ds_train, ds_test, confound)
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
    train_images, train_parents, test_images, test_parents, parent_dists, input_shape = dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_parents))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_parents))
    simulated_intervention_datasets = get_simulated_intervention_datasets(train_dataset, train_parents, parent_dists)

    encode_fn = get_encode_fn(parent_dims)
    train_dataset = train_dataset.map(encode_fn)
    test_dataset = test_dataset.map(encode_fn)


    def augment(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = layers.RandomZoom(height_factor=.2, width_factor=.2, fill_mode='constant', fill_value=-1.)(image)
        return image, parents

    train_data_dict = jax.tree_map(lambda ds: ds.map(augment), train_data_dict)

    if plot:
        for key, dataset in train_data_dict.items():
            show_images(dataset, f'train set {str(key)}')
        show_images(test_dataset, f'test set')

    return train_dataset, test_dataset, parent_dists, input_shape
