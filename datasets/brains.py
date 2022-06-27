import pickle
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
from numpy.typing import NDArray
from numpyro.distributions import Bernoulli
from PIL import Image
from staxplus import Shape
from tensorflow.keras import layers  # type: ignore
from tqdm import tqdm
from typing_extensions import Protocol
import jax.nn
from datasets.utils import ParentDist, Scenario, get_simulated_intervention_datasets
from utils import image_gallery

Array = NDArray[Any]


class ConfoundingFn(Protocol):
    def __call__(self, dataframe: pd.DataFrame, confound: bool) -> Tuple[Array, Dict[str, Array]]:
        ...


def image_thumbs(images: NDArray[np.uint8], parents: Dict[str, NDArray[Any]]) -> NDArray[np.uint8]:
    order = np.argsort(parents['sequence'], axis=-1)
    _image = image_gallery(images[order][..., np.newaxis], num_images_to_display=64, ncols=8, decode_fn=lambda x: x)
    _image = np.repeat(_image, repeats=3, axis=-1) if _image.shape[-1] == 1 else _image
    return _image


def get_dataset(dataset_dir: Path,
                confound: bool,
                confounding_fn: ConfoundingFn) -> Tuple[Array, Dict[str, Array], Array, Dict[str, Array]]:
    dataset_path = dataset_dir / 'data.pickle'
    try:
        with open(dataset_path, 'rb') as f1:
            train_images, train_parents, test_images, test_parents, = pickle.load(f1)
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

        def filter_fn(row: pd.Series) -> bool:
            return row.loc['t1_path'].exists() and row.loc['t2_flair_path'].exists()
        data = data[data.apply(filter_fn, axis=1)]
        # shuffle and split into train and test
        data = data.sample(frac=1., random_state=1)
        s = int(.7 * len(data))
        train_data, test_data = data.iloc[:s], data.iloc[s:]

        train_images, train_parents = confounding_fn(train_data, confound=confound)
        test_images, test_parents = confounding_fn(test_data, confound=False)
        with open(dataset_path, 'wb') as f2:
            pickle.dump((train_images, train_parents, test_images, test_parents), f2)
    return train_images, train_parents, test_images, test_parents


def read_image(path: Path) -> Array:
    with Image.open(path) as im:
        return np.array(im)


def sequence_age(data_dir: Path, confound: bool, confound_strength: float = 20.) \
        -> Tuple[str, Array, Dict[str, Array], Array, Dict[str, Array], Dict[str, ParentDist], Shape]:
    scenario_name = 'brains/sequence_age'
    dataset_name = 'confounded' if confound else 'unconfounded'
    dataset_dir = data_dir / scenario_name / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    k1, k2 = random.split(random.PRNGKey(1), 2)

    def sequence_age_model(age: Array) -> Array:
        p = (age - np.min(age)) / (np.max(age) - np.min(age))
        p = jax.nn.sigmoid(confound_strength * p - confound_strength / 2.)
        sequence = Bernoulli(probs=p).sample(k1)
        return np.array(sequence)

    def joint_plot(parents: Dict[str, NDArray[Any]]) -> sns.JointGrid:
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
        joint_plot({'age': age, 'sequence': sequence}).savefig(dataset_dir / 'this.png')
        images = np.array([read_image(row['t1_path'] if s else row['t2_flair_path'])
                           for s, (_, row) in zip(sequence, dataframe.iterrows())])
        parents = {'age': age, 'sequence': sequence}
        return images, parents

    train_images, train_parents, test_images, test_parents = get_dataset(dataset_dir, confound, confounding_fn)

    joint_plot(train_parents).savefig(dataset_dir / 'joint_hist_train.png')
    joint_plot(test_parents).savefig(dataset_dir / 'joint_hist_test.png')
    plt.imsave(str(dataset_dir / 'train_images.png'), image_thumbs(train_images, train_parents))
    plt.imsave(str(dataset_dir / 'test_images.png'), image_thumbs(test_images, test_parents))

    parent_dists \
        = {'sequence': ParentDist(name='sequence',
                                  dim=2,
                                  is_discrete=True,
                                  is_invertible=False,
                                  samples=train_parents['sequence']),
           'age': ParentDist(name='age',
                             dim=1,
                             is_discrete=False,
                             is_invertible=True,
                             samples=train_parents['age'])}
    input_shape = (-1, 28, 28, 3)
    return dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape


def brains(data_dir: Path, confound: bool, confound_strength: float) -> Tuple[str, Scenario]:

    dataset_name, train_images, train_parents, test_images, test_parents, parent_dists, input_shape \
        = sequence_age(data_dir, confound, confound_strength)

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

    augment_fn = layers.RandomTranslation(
        height_factor=(-.05, .05), width_factor=(-.1, .1), fill_mode='constant', fill_value=0.)

    def augment(image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        return augment_fn(image), parents

    train_data = tree_map(lambda ds: ds.map(augment).map(encode), train_data_dict)
    test_data = test_dataset.map(encode)

    return dataset_name, Scenario(train_data, test_data, parent_dists, input_shape, pmf)
