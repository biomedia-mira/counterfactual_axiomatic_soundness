
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.tree_util import tree_map
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from numpy.typing import NDArray
from numpyro.distributions import Bernoulli, Normal, TransformedDistribution, Uniform
from numpyro.distributions.transforms import AffineTransform, ComposeTransform, SigmoidTransform
from skimage import morphology, transform
from staxplus import Shape
from tensorflow.keras import layers  # type: ignore
from tqdm import tqdm

from datasets.morphomnist.measure import measure_image
from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import ConfoundingFn, Image, ParentDist, Scenario, get_simulated_intervention_datasets
from utils import image_gallery

Array = NDArray[Any]


def image_thumbs(images: NDArray[np.uint8], parents: Dict[str, NDArray[Any]]) -> NDArray[np.uint8]:
    order = np.argsort(parents['digit'], axis=-1)
    return image_gallery(images[order], num_images_to_display=128, decode_fn=lambda x: x)


def get_dataset(base_data_dir: Path,
                dataset_dir: Path,
                confound: bool,
                confounding_fn: ConfoundingFn) -> Tuple[Array, Dict[str, Array], Array, Dict[str, Array]]:
    dataset_path = dataset_dir / 'data.pickle'
    try:
        with open(dataset_path, 'rb') as f1:
            train_images, train_parents, test_images, test_parents, = pickle.load(f1)
    except FileNotFoundError:
        print('Dataset not found, creating new copy...')
        ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False,
                                      data_dir=f'{str(base_data_dir)}/mnist', as_supervised=True)
        train_images, train_parents = confounding_fn(ds_train, confound=confound)
        test_images, test_parents = confounding_fn(ds_test, confound=False)
        with open(dataset_path, 'wb') as f2:
            pickle.dump((train_images, train_parents, test_images, test_parents), f2)
    return train_images, train_parents, test_images, test_parents


@lru_cache(maxsize=None)
def _get_disk(radius: int, scale: int) -> Any:
    mag_radius = scale * radius
    mag_disk = morphology.disk(mag_radius, dtype=np.float64)  # type: ignore
    disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1, multichannel=False)
    return disk


def measure_thickness(image: Image) -> float:
    _image = image if image.shape[-1] == 1 else np.mean(image, axis=-1)
    return measure_image(_image).thickness


def set_thickness(image: Image, target_thickness: float) -> Image:
    morph = ImageMorphology(image[..., 0], scale=4)
    delta = target_thickness - morph.mean_thickness
    radius = int(morph.scale * abs(delta) / 2.)
    disk = _get_disk(radius, scale=4)
    img = morph.binary_image
    if delta >= 0:
        up_scale_image = morphology.dilation(img, disk)
    else:
        up_scale_image = morphology.erosion(img, disk)
    image = morph.downscale(np.float32(up_scale_image))
    return image[..., np.newaxis]


def meausure_intesity(image: Image) -> float:
    threshold = 0.5
    img_min, img_max = np.min(image), np.max(image)
    mask = (image >= img_min + (img_max - img_min) * threshold)
    avg_intensity = np.median(image[mask])
    return avg_intensity


def set_intensity(image: Image, intensity: float) -> Image:
    avg_intensity = meausure_intesity(image)
    factor = intensity / avg_intensity
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def measure_hue(image: Image) -> float:
    assert image.shape[-1] == 3
    threshold = 0.5
    hsv_image = rgb_to_hsv(image / 255.)
    hue_map = hsv_image[..., 0]
    brightness_map = hsv_image[..., -1]
    mask = brightness_map > threshold
    avg_hue = np.median(hue_map[mask]) * 360.
    return avg_hue


def set_hue(image: Image, hue: float) -> Image:
    hsv_image = rgb_to_hsv(image / 255.)
    hsv_image[..., 0] = hue / 360.  # set hue
    hsv_image[..., 1] = 1.  # set saturation to 1.
    return np.clip(hsv_to_rgb(hsv_image) * 255., 0, 255).astype(np.uint8)


# # Thickness intensity model
# def thickness_intensity_model(confound: bool,
#                               n_samples: Optional[int] = None,
#                               scale: float = 0.5,
#                               invert: bool = False) -> Tuple[Array, Array]:
#     with numpyro.plate('observations', n_samples):
#         k1, k2, k3 = random.split(random.PRNGKey(1), 3)
#         thickness_transform = ComposeTransform(
#             [AffineTransform(-1., 1.), SigmoidTransform(), AffineTransform(1.5, 4.5)])
#         thickness_dist = TransformedDistribution(Normal(0., 1.), thickness_transform)
#         thickness = cast(Array, numpyro.sample('thickness', thickness_dist, rng_key=k1))
#         multiplier = -1 if invert else 1
#         # if not confound intensity does not depend on thickness
#         _thickness = thickness if confound else cast(Array, numpyro.sample('thickness', thickness_dist, rng_key=k2))
#         loc = (_thickness - 2.5) * 2 * multiplier
#         transforms = ComposeTransform([SigmoidTransform(), AffineTransform(64, 191)])
#         intensity = numpyro.sample('intensity', TransformedDistribution(Normal(loc, scale), transforms), rng_key=k3)

#     thickness, intensity = np.array(thickness), np.array(intensity)
#     data = pd.DataFrame({'thickness': thickness, 'intensity': intensity})
#     sns.jointplot(data=data, x='thickness', y='intensity', kind="kde")
#     plt.show(block=False)
#     return thickness, intensity


def digit_hue(data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    scenario_name = 'digit_hue'
    dataset_name = f'confounded_scale_{scale:.2f}_outlier_prob_{outlier_prob:.3f}' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2, k3, k4 = random.split(random.PRNGKey(1), 4)

    def digit_hue_model(digit: Array) -> Array:
        sample_shape = (len(digit), )
        loc = (digit / 9. - .5) * 6.
        transforms = ComposeTransform([SigmoidTransform(), AffineTransform(0, 360.)])
        hue_dist = TransformedDistribution(Normal(loc, scale), transforms)
        mixture_probs = Bernoulli(probs=1.-outlier_prob).sample(k2, sample_shape)
        background_dist = Uniform(0., 360.)
        hue = hue_dist.sample(k3) * mixture_probs + (1. - mixture_probs) * background_dist.sample(k4, sample_shape)
        return np.array(hue)

    def confounding_fn(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        _digit = digit if confound else np.array(random.shuffle(k1, digit))
        hue = digit_hue_model(digit=_digit)
        images = np.array([set_hue(np.repeat(image, repeats=3, axis=-1), h)
                           for (image, _), h in tqdm(zip(dataset.as_numpy_iterator(), hue))])
        parents = {'digit': digit, 'hue': hue}
        return images, parents

    def joint_plot(parents: Dict[str, NDArray[Any]]) -> sns.JointGrid:
        data = pd.DataFrame({'digit': parents['digit'], 'hue': parents['hue']})
        grid = sns.JointGrid(x="digit", y="hue", data=data, ylim=(0, 360.), space=0)
        grid = grid.plot_joint(sns.boxplot, hue=np.zeros_like(data['digit']),
                               boxprops={'alpha': .5, 'edgecolor': 'black'},
                               flierprops={'marker': 'x'})
        grid.ax_joint.legend().remove()
        sns.histplot(x=parents['digit'], discrete=True, ax=grid.ax_marg_x)
        sns.kdeplot(y=parents['hue'], ax=grid.ax_marg_y, clip=(0.0, 360.), fill=True)
        return grid

    train_images, train_parents, test_images, test_parents \
        = get_dataset(data_dir, dataset_dir, confound, confounding_fn)

    joint_plot(train_parents).savefig(dataset_dir / 'joint_hist_train.png')
    joint_plot(test_parents).savefig(dataset_dir / 'joint_hist_test.png')
    plt.imsave(str(dataset_dir / 'train_images.png'), image_thumbs(train_images, train_parents))
    plt.imsave(str(dataset_dir / 'test_images.png'), image_thumbs(test_images, test_parents))

    parent_dists \
        = {'digit': ParentDist(name='digit',
                               dim=10,
                               is_discrete=True,
                               is_invertible=False,
                               samples=train_parents['digit']),
           'hue': ParentDist(name='hue',
                             dim=1,
                             is_discrete=False,
                             is_invertible=True,
                             samples=train_parents['hue'],
                             oracle=measure_hue)}
    input_shape = (-1, 28, 28, 3)
    return dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape


def confoudned_mnist(scenario_name: str, data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Scenario]:
    if scenario_name != 'digit_hue':
        raise NotImplementedError

    dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape \
        = digit_hue(data_dir, confound, scale, outlier_prob)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_parents))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_parents))
    train_data_dict, pmf = get_simulated_intervention_datasets(train_dataset, train_parents, parent_dists, num_bins=5)
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

    return dataset_name, Scenario(train_data, test_data, parent_dists, input_shape, pmf)
