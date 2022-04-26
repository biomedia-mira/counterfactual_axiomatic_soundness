import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.tree_util import tree_map
from models.utils import ParentDist
from numpyro.distributions import Normal, TransformedDistribution
from numpyro.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from skimage import morphology, transform
from staxplus import Shape
from tensorflow.keras import layers
from tqdm import tqdm

from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import IMAGE, Array, ConfoundingFn, Scenario, get_simulated_intervention_datasets
from utils import image_gallery


def get_dataset(data_dir: Path, dataset_name: str, confound: bool, confounding_fn: ConfoundingFn) \
        -> Tuple[Array, Dict[str, Array], Array, Dict[str, Array]]:
    dataset_path = Path(f'{str(data_dir)}/{dataset_name}' + ('_confounded' if confound else ''))
    try:
        with open(dataset_path, 'rb') as f1:
            train_images, train_parents, test_images, test_parents, = pickle.load(f1)
    except FileNotFoundError:
        print('Dataset not found, creating new copy...')
        ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False,
                                      data_dir=f'{str(data_dir)}/mnist', as_supervised=True)
        train_images, train_parents = confounding_fn(ds_train, confound=confound)
        test_images, test_parents = confounding_fn(ds_test, confound=False)
        with open(dataset_path, 'wb') as f2:
            pickle.dump((train_images, train_parents, test_images, test_parents), f2)
    return train_images, train_parents, test_images, test_parents


def set_colour(image: IMAGE, colour: Array) -> IMAGE:
    return np.array(np.repeat(image, 3, axis=-1) * colour).astype(np.uint8)


@lru_cache(maxsize=None)
def _get_disk(radius: int, scale: int) -> Any:
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
def thickness_intensity_model(confound: bool,
                              n_samples: Optional[int] = None,
                              scale: float = 0.5,
                              invert: bool = False) -> Tuple[Array, Array]:
    with numpyro.plate('observations', n_samples):
        k1, k2, k3 = random.split(random.PRNGKey(1), 3)
        thickness_transform = ComposeTransform(
            [AffineTransform(-1., 1.), SigmoidTransform(), AffineTransform(1.5, 4.5)])
        thickness_dist = TransformedDistribution(Normal(0., 1.), thickness_transform)
        thickness = numpyro.sample('thickness', thickness_dist, rng_key=k1)
        multiplier = -1 if invert else 1
        # if not confound intensity does not depend on thickness
        _thickness = thickness if confound else numpyro.sample('thickness', thickness_dist, rng_key=k2)
        loc = (_thickness - 2.5) * 2 * multiplier
        transforms = ComposeTransform([SigmoidTransform(), AffineTransform(64, 191)])
        intensity = numpyro.sample('intensity', TransformedDistribution(Normal(loc, scale), transforms), rng_key=k3)
    return np.array(thickness), np.array(intensity)


def digit_thickness_intensity(data_dir: Path, confound: bool) \
        -> Tuple[Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    def confounding_fn(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        thickness, intensity = thickness_intensity_model(confound=confound, n_samples=len(dataset))
        images = np.array([set_intensity(set_thickness(image, t), i)
                           for (image, _), t, i in tqdm(zip(dataset.as_numpy_iterator(), thickness, intensity))])
        parents = {'digit': digit, 'thickness': thickness, 'intensity': intensity}
        return images, parents

    dataset_name = 'digit_thickness_intensity'
    train_images, train_parents, test_images, test_parents \
        = get_dataset(data_dir, dataset_name, confound, confounding_fn)

    parent_dists \
        = {'digit': ParentDist(name='digit',
                               dim=10,
                               is_discrete=True,
                               is_invertible=False,
                               samples=train_parents['digit']),
           'thickness': ParentDist(name='thickness',
                                   dim=1,
                                   is_discrete=False,
                                   is_invertible=True,
                                   samples=train_parents['thickness']),
           'intensity': ParentDist(name='intensity',
                                   dim=1,
                                   is_discrete=False,
                                   is_invertible=True,
                                   samples=train_parents['intensity'])}
    input_shape = (-1, 28, 28, 1)

    return train_images, train_parents, test_images, test_parents, parent_dists, input_shape


def show_images(dataset: tf.data.Dataset, title: str) -> None:
    data = iter(dataset.batch(128)).__next__()
    order = np.argsort(np.argmax(data[1]['digit'], axis=-1))
    plt.imshow(image_gallery(data[0].numpy()[order], num_images_to_display=128))
    plt.title(title)
    plt.show(block=False)


def confounded_mnist(data_dir: Path, dataset_name: str, confound: bool, plot: bool = False) -> Scenario:
    data_fn = {'digit_thickness_intensity': digit_thickness_intensity}[dataset_name]

    train_images, train_parents, test_images, test_parents, parent_dists, input_shape = data_fn(data_dir, confound)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_parents))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_parents))
    train_data_dict = get_simulated_intervention_datasets(train_dataset, train_parents, parent_dists)
    if not confound:
        train_data_dict = {key: train_dataset for key in train_data_dict.keys()}

    def encode(image: tf.Tensor, patents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = (tf.cast(image, tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        patents = {parent: tf.one_hot(value, parent_dists[parent].dim) if parent_dists[parent].is_discrete
                   else tf.expand_dims(value, axis=-1) for parent, value in patents.items()}
        return image, patents

    def augment(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = layers.RandomZoom(height_factor=.2, width_factor=.2, fill_mode='constant', fill_value=0.)(image)
        return image, parents

    train_data = tree_map(lambda ds: ds.map(augment).map(encode), train_data_dict)
    test_data = test_dataset.map(encode)

    if plot:
        for key, dataset in train_data_dict.items():
            show_images(dataset, f'train set {str(key)}')
        show_images(test_data, 'test set')

    return train_data, test_data, parent_dists, input_shape
