import math
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
import jax.numpy as jnp
from datasets.morphomnist.measure import measure_image
from datasets.morphomnist.morpho import ImageMorphology
from datasets.utils import ConfoundingFn, Image, ParentDist, Scenario, get_simulated_intervention_datasets
from utils import image_gallery

Array = NDArray[Any]


def image_thumbs(images: NDArray[np.uint8], parents: Dict[str, NDArray[Any]]) -> NDArray[np.uint8]:
    order = np.argsort(parents['digit'], axis=-1)
    _image = image_gallery(images[order], num_images_to_display=128, decode_fn=lambda x: x)
    _image = np.repeat(_image, repeats=3, axis=-1) if _image.shape[-1] == 1 else _image
    return _image


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
    avg_hue = np.median(hue_map[mask])
    return avg_hue


def set_hue_and_saturation(image: Image, hue: float, saturation: float = 1.) -> Image:
    hsv_image = rgb_to_hsv(image / 255.)
    hsv_image[..., 0] = hue  # set hue
    hsv_image[..., 1] = saturation
    return np.clip(hsv_to_rgb(hsv_image) * 255., 0, 255).astype(np.uint8)

##


def digit_thickness(data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    scenario_name = 'digit_thickness'
    dataset_name = f'confounded_scale_{scale:.2f}_outlier_prob_{outlier_prob:.3f}' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2, k3, k4 = random.split(random.PRNGKey(1), 4)

    def digit_thickness(digit: Array) -> Array:
        sample_shape = (len(digit), )
        loc = (digit / 10.) * 4.5 + 1.5
        thickness_dist = Normal(loc, scale)
        mixture_probs = Bernoulli(probs=1. - outlier_prob).sample(k1, sample_shape)
        background_dist = Uniform(1.5, 6.)
        thickness = thickness_dist.sample(k2) * mixture_probs + (1. - mixture_probs) * \
            background_dist.sample(k3, sample_shape)
        return np.array(thickness)

    def confounding_fn(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        _digit = digit if confound else np.array(random.shuffle(k4, digit))
        thickness = digit_thickness(digit=_digit)
        images = np.array([set_thickness(image, t)
                          for (image, _), t in tqdm(zip(dataset.as_numpy_iterator(), thickness))])
        parents = {'digit': digit, 'thickness': thickness}
        return images, parents

    def joint_plot(parents: Dict[str, NDArray[Any]]) -> sns.JointGrid:
        data = pd.DataFrame({'digit': parents['digit'], 'thickness': parents['thickness']})
        grid = sns.JointGrid(x="digit", y="thickness", data=data, ylim=(1.5, 6.), space=0)
        grid = grid.plot_joint(sns.boxplot, hue=np.zeros_like(data['digit']),
                               boxprops={'alpha': .5, 'edgecolor': 'black'},
                               flierprops={'marker': 'x'})
        grid.ax_joint.legend().remove()
        sns.histplot(x=parents['digit'], discrete=True, ax=grid.ax_marg_x)
        sns.kdeplot(y=parents['thickness'], ax=grid.ax_marg_y, clip=(1.5, 6.), fill=True)
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
            'thickness': ParentDist(name='thickness',
                                    dim=1,
                                    is_discrete=False,
                                    is_invertible=True,
                                    samples=train_parents['thickness'],
                                    oracle=measure_thickness)}
    input_shape = (-1, 28, 28, 1)
    return dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape


def thickness_hue(data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    scenario_name = 'thickness_hue'
    dataset_name = f'confounded_scale_{scale:.2f}_outlier_prob_{outlier_prob:.3f}' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2, k3, k4, k5 = random.split(random.PRNGKey(1), 5)

    def thickness_hue_model(n_samples: int, _confound: bool) -> Tuple[Array, Array]:
        sample_shape = (n_samples, )
        thickness_dist = Uniform(1.5, 6.)
        thickness = thickness_dist.sample(k1, sample_shape)
        # if not confound hue does not depend on thickness
        _thickness = thickness if _confound else thickness_dist.sample(k2, sample_shape)
        loc = (_thickness - 1.5) / 4.5
        hue_dist = Normal(loc, scale)
        mixture_probs = Bernoulli(probs=1. - outlier_prob).sample(k3, sample_shape)
        background_dist = Uniform(0., 1.)
        hue = hue_dist.sample(k4) * mixture_probs + (1. - mixture_probs) * background_dist.sample(k5, sample_shape)
        hue = jnp.clip(hue, a_min=0., a_max=1.)
        return np.array(thickness), np.array(hue)

    def confounding_fn(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        thickness, hue = thickness_hue_model(n_samples=digit.shape[0], _confound=confound)
        images = np.array([set_hue_and_saturation(np.repeat(set_thickness(image, t), repeats=3, axis=-1), h)
                           for (image, _), h, t in tqdm(zip(dataset.as_numpy_iterator(), hue, thickness))])
        parents = {'digit': digit, 'thickness': thickness, 'hue': hue}
        return images, parents

    def joint_plot(parents: Dict[str, NDArray[Any]]) -> sns.JointGrid:
        data = pd.DataFrame({'thickness': parents['thickness'], 'hue': parents['hue']})
        g = sns.jointplot(data=data, x='thickness', y='hue', kind="kde", ylim=(0, 1.), xlim=(1.5, 6.))
        g.plot_joint(sns.scatterplot)
        return g

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
            'thickness': ParentDist(name='thickness',
                                    dim=1,
                                    is_discrete=False,
                                    is_invertible=True,
                                    samples=train_parents['thickness'],
                                    oracle=measure_thickness),
           'hue': ParentDist(name='hue',
                             dim=1,
                             is_discrete=False,
                             is_invertible=True,
                             samples=train_parents['hue'],
                             oracle=measure_hue)}
    input_shape = (-1, 28, 28, 3)
    return dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape


def digit_hue(data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    scenario_name = 'digit_hue'
    dataset_name = f'confounded_scale_{scale:.2f}_outlier_prob_{outlier_prob:.3f}' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2, k3, k4 = random.split(random.PRNGKey(1), 4)

    def digit_hue_model(digit: Array) -> Array:
        sample_shape = (len(digit), )
        loc = digit / 10. + .05
        hue_dist = Normal(loc, scale)
        mixture_probs = Bernoulli(probs=1. - outlier_prob).sample(k1, sample_shape)
        background_dist = Uniform(0., 1.)
        hue = hue_dist.sample(k2) * mixture_probs + (1. - mixture_probs) * background_dist.sample(k3, sample_shape)
        return np.array(hue)

    def confounding_fn(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        _digit = digit if confound else np.array(random.shuffle(k4, digit))
        hue = digit_hue_model(digit=_digit)
        images = np.array([set_hue_and_saturation(np.repeat(image, repeats=3, axis=-1), h)
                           for (image, _), h in tqdm(zip(dataset.as_numpy_iterator(), hue))])
        parents = {'digit': digit, 'hue': hue}
        return images, parents

    def joint_plot(parents: Dict[str, NDArray[Any]]) -> sns.JointGrid:
        data = pd.DataFrame({'digit': parents['digit'], 'hue': parents['hue']})
        grid = sns.JointGrid(x="digit", y="hue", data=data, ylim=(0, 1.), space=0)
        grid = grid.plot_joint(sns.boxplot, hue=np.zeros_like(data['digit']),
                               boxprops={'alpha': .5, 'edgecolor': 'black'},
                               flierprops={'marker': 'x'})
        grid.ax_joint.legend().remove()
        sns.histplot(x=parents['digit'], discrete=True, ax=grid.ax_marg_x)
        sns.kdeplot(y=parents['hue'], ax=grid.ax_marg_y, clip=(0.0, 1.), fill=True)
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


def digit_hue_saturation(data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    scenario_name = 'digit_hue_saturation'
    dataset_name = f'confounded_scale_{scale:.2f}_outlier_prob_{outlier_prob:.3f}' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2, k3, k4, k5, k6, k7, k8 = random.split(random.PRNGKey(1), 8)

    def digit_hue_saturation_model(digit: Array, confound: bool = True) -> Tuple[Array, Array]:
        sample_shape = (len(digit), )
        # hue
        loc = digit / 10. + .05
        loc = loc if confound else random.permutation(k1, loc, independent=True)
        hue_dist = Normal(loc, scale)
        is_inlier = Bernoulli(probs=1. - outlier_prob).sample(k2, sample_shape)
        background_dist = Uniform(0., 1.)
        hue = hue_dist.sample(k3) * is_inlier + (1. - is_inlier) * background_dist.sample(k4, sample_shape)
        # saturation
        loc = hue * .7 + .3
        loc = loc if confound else random.shuffle(k5, loc)
        saturation_dist = Normal(loc, scale)
        is_inlier = Bernoulli(probs=1. - outlier_prob).sample(k6, sample_shape)
        background_dist = Uniform(.3, 1.)
        saturation = saturation_dist.sample(k7) * is_inlier + (1. - is_inlier) * background_dist.sample(k8, sample_shape)
        return np.array(hue), np.array(saturation)

    def confounding_fn(dataset: tf.data.Dataset, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        digit = np.array([digit for _, digit in iter(dataset)])
        hue, saturation = digit_hue_saturation_model(digit=digit, confound=confound)
        images = np.array([set_hue_and_saturation(np.repeat(image, repeats=3, axis=-1), h, s)
                           for (image, _), h, s in tqdm(zip(dataset.as_numpy_iterator(), hue, saturation))])
        parents = {'digit': digit, 'hue': hue, 'saturation': saturation}
        return images, parents

    def joint_plot(parents: Dict[str, NDArray[Any]]) -> Tuple[sns.JointGrid, sns.JointGrid]:
        data = pd.DataFrame({'digit': parents['digit'], 'hue': parents['hue']})
        g1 = sns.JointGrid(x="digit", y="hue", data=data, ylim=(0, 1.), space=0)
        g1 = g1.plot_joint(sns.boxplot, hue=np.zeros_like(data['digit']),
                           boxprops={'alpha': .5, 'edgecolor': 'black'},
                           flierprops={'marker': 'x'})
        g1.ax_joint.legend().remove()
        sns.histplot(x=parents['digit'], discrete=True, ax=g1.ax_marg_x)
        sns.kdeplot(y=parents['hue'], ax=g1.ax_marg_y, clip=(0.0, 1.), fill=True)
        #
        data = pd.DataFrame({'hue': parents['hue'], 'saturation': parents['saturation']})
        g2 = sns.jointplot(x="hue", y="saturation", data=data, xlim=(0., 1.), ylim=(0, 1.), kind='kde')
        g2.plot_joint(sns.scatterplot)
        g2.ax_joint.legend().remove()
        return g1, g2

    train_images, train_parents, test_images, test_parents \
        = get_dataset(data_dir, dataset_dir, confound, confounding_fn)

    g1, g2 = joint_plot(train_parents)
    g1.savefig(dataset_dir / 'joint_hist_digt_hue_train.png')
    g2.savefig(dataset_dir / 'joint_hist_hue_saturation_train.png')
    g1, g2 = joint_plot(test_parents)
    g1.savefig(dataset_dir / 'joint_hist_digit_hue_test.png')
    g2.savefig(dataset_dir / 'joint_hist_hue_saturation_test.png')
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
                             oracle=measure_hue),
           'saturation': ParentDist(name='saturation',
                                    dim=1,
                                    is_discrete=False,
                                    is_invertible=True,
                                    samples=train_parents['saturation'])}
    input_shape = (-1, 28, 28, 3)
    return dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape


def confoudned_mnist(scenario_name: str, data_dir: Path, confound: bool, scale: float, outlier_prob: float) \
        -> Tuple[str, Scenario]:

    scenario_fns = {'digit_hue': digit_hue,
                    'digit_thickness': digit_thickness,
                    'thickness_hue': thickness_hue,
                    'digit_hue_saturation': digit_hue_saturation
                    }
    if scenario_name not in scenario_fns.keys():
        raise NotImplementedError
    else:
        scenario_fn = scenario_fns[scenario_name]
    dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape \
        = scenario_fn(data_dir, confound, scale, outlier_prob)

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

    # augment_fn = tf.keras.Sequential([
    #     layers.RandomRotation(factor=(-.1, .1), fill_mode='constant', fill_value=0.),
    #     layers.RandomTranslation(height_factor=(-.1, .1), width_factor=(-.1, .1), fill_mode='constant', fill_value=0.),
    #     layers.RandomZoom(height_factor=(-.1, .1), width_factor=(-.1, .1), fill_mode='constant', fill_value=0.)])

    augment_fn = layers.RandomTranslation(
        height_factor=(-.05, .05), width_factor=(-.1, .1), fill_mode='constant', fill_value=0.)

    def augment(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        return augment_fn(image), parents

    train_data = tree_map(lambda ds: ds.map(augment).map(encode), train_data_dict)
    test_data = test_dataset.map(encode)

    return dataset_name, Scenario(train_data, test_data, parent_dists, input_shape, pmf)
