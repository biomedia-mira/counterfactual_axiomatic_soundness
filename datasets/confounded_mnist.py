from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.typing import NDArray

from components.stax_extension import Shape
from datasets.utils import Mechanism, get_marginal_datasets, image_gallery, load_cached_dataset


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
                                    train_mechanisms: List[Mechanism],
                                    test_mechanisms: List[Mechanism],
                                    parent_dims: Dict[str, int]) \
        -> Tuple[Dict[FrozenSet[str], tf.data.Dataset], tf.data.Dataset, Dict[str, NDArray], Shape]:
    tf.config.experimental.set_visible_devices([], 'GPU')
    input_shape = (-1, 28, 28, 3)
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=False, data_dir=f'{data_dir}/mnist')

    encode_fn = get_encode_fn(parent_dims)
    dataset_dir = Path(f'{data_dir}/confounded_mnist')
    train_data, train_parents = load_cached_dataset(dataset_dir / 'train', ds_train, train_mechanisms, parent_dims)
    train_data = train_data.map(encode_fn)

    test_data, _ = load_cached_dataset(dataset_dir / 'test', ds_test, test_mechanisms, parent_dims)
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
