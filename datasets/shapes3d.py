from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from more_itertools import powerset
from tqdm import tqdm

from datasets.utils import ParentDist, Scenario


def shapes3d(data_dir: Path, batch_size: int) -> Tuple[str, Scenario]:
    parent_names = ('label_floor_hue',
                    'label_object_hue',
                    'label_orientation',
                    'label_scale',
                    'label_shape',
                    'label_wall_hue')

    input_shape = (-1, 64, 64, 3)
    ds = tfds.load('shapes3d', split='train', shuffle_files=False, data_dir=f'{str(data_dir)}/shapes3d')
    ds = ds.shuffle(buffer_size=len(ds), seed=1)
    s = int(.90 * len(ds))
    ds_train = ds.take(s)
    ds_test = ds.skip(s)

    # build parent distributions for endogenous and exogenous parents
    train_parents = {key: [] for key in parent_names}
    for data in tqdm(ds_train.take(50000).batch(1000, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)):
        for parent_name in train_parents.keys():
            train_parents[parent_name] += list(data[parent_name])
    train_parents = {key: np.array(value) for key, value in train_parents.items()}

    parent_dists: Dict[str, ParentDist] = {}
    for parent_name in parent_names:
        if parent_name == 'label_shape':
            dim = 4
        elif parent_name == 'label_orientation':
            dim = 15
        else:
            dim = 10
        parent_dists[parent_name] = \
            ParentDist(name=parent_name,
                       dim=dim,
                       is_discrete=True,
                       samples=train_parents[parent_name])

    @tf.function
    def encode(data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        image = (tf.cast(data['image'], tf.float32) - tf.constant(127.5)) / tf.constant(127.5)
        parents = {p_name: tf.one_hot(data[p_name], parent_dists[p_name].dim)
                   if parent_dists[p_name].is_discrete
                   else tf.expand_dims(data[p_name], axis=-1)
                   for p_name in parent_names}
        return image, parents

    ds_train = ds_train.map(encode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(encode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_data = {frozenset(key): ds_train for key in powerset(parent_names)}
    test_data_uc = ds_test
    test_data_c = ds_test
    return 'shapes3d', Scenario(train_data, test_data_uc, test_data_c, parent_dists, input_shape, None)
