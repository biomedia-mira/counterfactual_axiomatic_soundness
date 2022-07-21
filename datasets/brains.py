import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import jax.nn
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from jax.tree_util import tree_map
from numpy.typing import NDArray
from numpyro.distributions import Bernoulli
from PIL import Image
from tensorflow.keras import layers  # type: ignore
from typing_extensions import Protocol

from datasets.utils import ParentDist, Scenario, get_simulated_intervention_datasets
from utils import image_gallery

Array = NDArray[Any]


class ConfoundingFn(Protocol):
    def __call__(self, dataframe: pd.DataFrame, confound: bool) -> Tuple[Array, Dict[str, Array]]:
        ...


def save_image_thumbs(path: str, images: NDArray[np.uint8], parents: Dict[str, NDArray[Any]]) -> None:
    order = np.argsort(parents['sequence'], axis=-1)
    _image = image_gallery(images[order], num_images_to_display=64, ncols=8, decode_fn=lambda x: x)
    _image = np.repeat(_image, repeats=3, axis=-1) if _image.shape[-1] == 1 else _image
    plt.imsave(path, _image)
    return


def get_dataset(dataset_dir: Path,
                confound: bool,
                confounding_fn: ConfoundingFn,
                joint_plot_fn: Callable[[Dict[str, Array]], sns.JointGrid]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Array]]:
    dataset_path = dataset_dir / 'data.pickle'
    try:
        with open(dataset_path, 'rb') as f1:
            (train_images, train_parents,
             test_images_uc, test_parents_uc,
             test_images_c, test_parents_c) = pickle.load(f1)
    except FileNotFoundError:
        print('Dataset not found, creating new copy...')

        thumb_directory = Path('/vol/biodata/data/biobank/12579/brain/t0/rigid_to_mni/thumbs')
        csv_path = Path('/vol/biodata/data/biobank/12579/brain/ukb21079_extracted.csv')
        full_data = pd.read_csv(csv_path, index_col='eid')
        data = pd.DataFrame({'sex': full_data['31-0.0'],
                             'age': full_data['21003-2.0'],
                             't1_path': full_data.index.map(lambda eid: thumb_directory / f'{eid}_T1_unbiased_brain_rigid_to_mni.png'),
                             't2_flair_path': full_data.index.map(lambda eid: thumb_directory / f'{eid}_T2_FLAIR_unbiased_brain_rigid_to_mni.png')})
        # clean the data, drop nans and none existing images.
        data = data.dropna()

        def filter_fn(row: pd.DataFrame) -> bool:
            return row.loc['t1_path'].exists() and row.loc['t2_flair_path'].exists()
        data = data[data.apply(filter_fn, axis=1)]
        # shuffle and split into train and test
        data = data.sample(frac=1., random_state=1)
        s = int(.7 * len(data))
        train_data, test_data = data.iloc[:s], data.iloc[s:]

        train_images, train_parents = confounding_fn(train_data, confound=confound)
        test_images_uc, test_parents_uc = confounding_fn(test_data, confound=False)
        test_images_c, test_parents_c = confounding_fn(test_data, confound=True)

        with open(dataset_path, 'wb') as f2:
            _data = (train_images, train_parents,
                     test_images_uc, test_parents_uc,
                     test_images_c, test_parents_c)
            pickle.dump(_data, f2)

    joint_plot_fn(train_parents).savefig(dataset_dir / 'joint_hist_train.png')
    joint_plot_fn(test_parents_uc).savefig(dataset_dir / 'joint_hist_test_unconfounded.png')
    joint_plot_fn(test_parents_c).savefig(dataset_dir / 'joint_hist_test_confounded.png')

    save_image_thumbs(str(dataset_dir / 'train_images.png'), train_images, train_parents)
    save_image_thumbs(str(dataset_dir / 'unconfounded_test_images.png'), test_images_uc, test_parents_uc)
    save_image_thumbs(str(dataset_dir / 'confounded_test_images.png'), test_images_c, test_parents_c)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_parents))
    test_dataset_uc = tf.data.Dataset.from_tensor_slices((test_images_uc, test_parents_uc))
    test_dataset_c = tf.data.Dataset.from_tensor_slices((test_images_c, test_parents_c))

    return train_dataset, test_dataset_uc, test_dataset_c, train_parents


def read_image(path: Path) -> Array:
    with Image.open(path) as im:
        return np.expand_dims(np.array(im), axis=-1)


def sequence_age(data_dir: Path, confound: bool, confound_strength: float = 20.) \
        -> Tuple[str, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, ParentDist]]:
    scenario_name = 'brains/sequence_age'
    dataset_name = f'confounded_confound_strength_{confound_strength:.1f}' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2 = random.split(random.PRNGKey(1), 2)

    def sequence_age_model(age: Array) -> Array:
        p = (age - np.min(age)) / (np.max(age) - np.min(age))
        p = jax.nn.sigmoid(confound_strength * p - confound_strength / 2.)
        sequence = Bernoulli(probs=p).sample(k1)
        return np.array(sequence)

    def joint_plot_fn(parents: Dict[str, NDArray[Any]]) -> sns.JointGrid:
        data = pd.DataFrame({'sequence': parents['sequence'], 'age': parents['age']})
        grid = sns.JointGrid(x="sequence", y="age", data=data, space=0)
        grid = grid.plot_joint(sns.boxplot, hue=np.zeros_like(data['sequence']),
                               boxprops={'alpha': .5, 'edgecolor': 'black'},
                               flierprops={'marker': 'x', 'zorder': -1,
                                           'markerfacecolor': 'black',
                                           'markeredgecolor': 'black'})
        grid.ax_joint.legend().remove()
        sns.histplot(x=parents['sequence'], discrete=True, ax=grid.ax_marg_x)
        sns.kdeplot(y=parents['age'], ax=grid.ax_marg_y, fill=True, bw_adjust=1.5)
        return grid

    def confounding_fn(dataframe: pd.DataFrame, confound: bool = True) -> Tuple[Array, Dict[str, Array]]:
        age = np.array(dataframe['age'])
        _age = age if confound else np.array(random.shuffle(k2, age))
        sequence = sequence_age_model(age=_age)
        images = np.array([read_image(row['t1_path'] if s else row['t2_flair_path'])
                           for s, (_, row) in zip(sequence, dataframe.iterrows())])
        parents = {'age': age, 'sequence': sequence}
        return images, parents

    train_dataset, test_dataset_uc, test_dataset_c, train_parents = get_dataset(
        dataset_dir, confound, confounding_fn, joint_plot_fn)

    parent_dists \
        = {'sequence': ParentDist(name='sequence',
                                  dim=2,
                                  is_discrete=True,
                                  samples=train_parents['sequence']),
           'age': ParentDist(name='age',
                             dim=1,
                             is_discrete=False,
                             samples=train_parents['age'])}

    return dataset_name, train_dataset, test_dataset_uc, test_dataset_c, parent_dists


def brains(data_dir: Path, confound: bool, confound_strength: float, batch_size: int) -> Tuple[str, Scenario]:
    dataset_name, train_dataset, test_dataset_uc, test_dataset_c, parent_dists \
        = sequence_age(data_dir, confound, confound_strength)

    @tf.function
    def encode(image: tf.Tensor, patents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = tf.image.resize(image[18:214, :196], size=(64, 64))
        image = (tf.cast(image, tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        patents = {parent: tf.one_hot(value, parent_dists[parent].dim) if parent_dists[parent].is_discrete
                   else tf.expand_dims(value, axis=-1) for parent, value in patents.items()}
        return image, patents

    augment_fn = tf.keras.Sequential([
        layers.RandomTranslation(height_factor=(-.1, .1), width_factor=(-.1, .1), fill_mode='constant', fill_value=-1.),
        layers.RandomZoom(height_factor=(-.1, .1), width_factor=(-.1, .1), fill_mode='constant', fill_value=-1.)])

    @tf.function
    def augment(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        return augment_fn(image), parents

    train_dataset = train_dataset.map(encode, num_parallel_calls=tf.data.AUTOTUNE)

    train_data_dict, pmf = get_simulated_intervention_datasets(train_dataset, parent_dists, num_bins=5, cache=True)
    if not confound:
        train_data_dict = {key: train_dataset.cache() for key in train_data_dict.keys()}

    train_data = tree_map(lambda ds: ds.repeat().
                          batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE).
                          map(augment, num_parallel_calls=tf.data.AUTOTUNE).
                          prefetch(tf.data.AUTOTUNE), train_data_dict)

    test_data_uc = test_dataset_uc.map(encode, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_data_c = test_dataset_c.map(encode, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(
        batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    input_shape = (-1, 64, 64, 1)
    return dataset_name, Scenario(train_data, test_data_uc, test_data_c, parent_dists, input_shape, pmf)
