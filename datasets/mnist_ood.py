from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.confounded_mnist import get_colourise_fn, get_encode_fn, show_images
from datasets.utils import get_uniform_confusion_matrix
from datasets.utils import load_cached_dataset


def get_coloured_kmnist(data_dir: Path, plot: bool = False) -> tf.data.Dataset:
    dataset_name = 'coloured_kmnist'
    parent_dims = {'digit': 10, 'colour': 10}

    _, ds_test = tfds.load('kmnist', split=['train', 'test'], shuffle_files=False,
                           data_dir=f'{str(data_dir)}/kmnist', as_supervised=True)

    dataset_dir = Path(f'{str(data_dir)}/{dataset_name}')
    colourise_fn = get_colourise_fn(get_uniform_confusion_matrix(10, 10))
    confounding_fns = [colourise_fn]
    test_data, _ = load_cached_dataset(dataset_dir / 'test', ds_test, confounding_fns, parent_dims)
    encode_fn = get_encode_fn(parent_dims)
    test_data = test_data.map(encode_fn)

    if plot:
        show_images(test_data, f'test set')

    return test_data
