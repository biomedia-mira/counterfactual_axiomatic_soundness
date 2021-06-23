import itertools
from typing import Tuple, Dict, Callable, List, Generator, Any
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets.jpeg import get_jpeg_encode_decode_fns
MechanismFn = Callable[[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]], Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]]


def get_uniform_confusion_matrix(num_rows: int, num_columns: int) -> np.ndarray:
    return np.ones((num_rows, num_columns)) / num_columns


def get_random_confusion_matrix(num_rows: int, num_columns: int, temperature: float = .1, seed: int = 1) -> np.ndarray:
    random_state = np.random.RandomState(seed=seed)
    logits = random_state.random(size=(num_rows, num_columns))
    tmp = np.exp(logits / temperature)
    return tmp / tmp.sum(1, keepdims=True)


def get_diagonal_confusion_matrix(num_rows: int, num_columns: int, noise: float = 0.) -> np.ndarray:
    assert num_rows == num_columns
    return (np.eye(num_rows) * (1. - noise)) + (np.ones((num_rows, num_rows)) - np.eye(num_rows)) * noise / (
            num_rows - 1)


def get_colorize_fn(cm: np.ndarray, labels: np.ndarray) -> MechanismFn:
    color_indices = tf.convert_to_tensor(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), labels)),
                                         dtype=tf.int64)
    colors = tf.convert_to_tensor(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
                                   (.5, 0, 0), (0, .5, 0), (0, 0, .5)))

    def colorize_fn(index: tf.Tensor, image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> Tuple[
        tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        color_idx = color_indices[index]
        parents.update({'color': color_idx})
        return index, tf.concat([image] * 3, axis=-1) * tf.reshape(colors[color_idx], (1, 1, 3)), parents

    return colorize_fn


def apply_mechanisms_to_dataset(dataset: tf.data.Dataset, mechanisms: List[MechanismFn]) -> tf.data.Dataset:
    dataset = dataset.map(lambda image, label: (tf.cast(image, tf.float32), {'digit': label}))
    dataset = dataset.enumerate().map(lambda index, data: (index, data[0], data[1]))
    for mechanism in mechanisms:
        dataset = dataset.map(mechanism, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def create_uncounfounded_datasets(dataset: tf.data.Dataset, parent_dims: Dict[str, int]) -> Tuple[
    Dict[str, tf.data.Dataset], Dict[str, np.ndarray]]:
    tmp = [item[-1] for item in dataset.as_numpy_iterator()]
    parents = {key: np.array([item[key] for item in tmp]) for key in tmp[0].keys()}
    indicator = {key: [parents[key] == i for i in range(dim)] for key, dim in parent_dims.items()}
    index_map = np.array([np.logical_and.reduce(a) for a in itertools.product(*indicator.values())])
    index_map = index_map.reshape((*parent_dims.values(), -1))
    counts = np.sum(index_map, axis=-1)
    joint_distribution = counts / np.sum(counts)
    datasets, marginals = {}, {}

    for axis, parent in enumerate(parents.keys()):
        marginal_distribution = np.sum(joint_distribution, axis=tuple(set(range(counts.ndim)) - {axis}), keepdims=True)
        distribution = marginal_distribution * np.sum(joint_distribution, axis=axis, keepdims=True)
        weights = distribution / counts
        num_repeats = tf.convert_to_tensor(np.round(weights / np.min(weights)).astype(int))

        def oversample(index: tf.Tensor, image: tf.Tensor, parents: Dict[str, tf.Tensor]) -> tf.data.Dataset:
            return tf.data.Dataset.from_tensors((index, image, parents)).repeat(num_repeats[list(parents.values())])

        datasets[parent] = dataset.flat_map(oversample)
        marginals[parent] = np.squeeze(marginal_distribution)

    return datasets, marginals


def prepare_dataset(dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=60000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    # encode_fn, decode_fn = get_jpeg_encode_decode_fns(max_seq_len=300, block_size=(8, 8))
    # dataset = dataset.apply(lambda x: encode_fn(x)[0])
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return dataset


def create_confounded_mnist_dataset(batch_size: int):
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)
    parent_dims = {'digit': 10, 'color': 10}
    train_targets = np.array([y for x, y in ds_train.as_numpy_iterator()])
    test_targets = np.array([y for x, y in ds_train.as_numpy_iterator()])

    colorize_cm = get_diagonal_confusion_matrix(10, 10, noise=.1)
    colorize_fun = get_colorize_fn(colorize_cm, train_targets)

    ds_train = apply_mechanisms_to_dataset(ds_train, [colorize_fun])
    unconfounded_datasets, marginals = create_uncounfounded_datasets(ds_train, parent_dims)

    ds = tf.data.Dataset.zip({'joint': ds_train, **unconfounded_datasets})

    ds = prepare_dataset(ds, batch_size)

    for i, j, k in ds:
        print('as')

    # def data_stream() -> Generator[Any, None, None]:
    #     for inputs in tqdm(zip(ds_train, *unconfounded_datasets.values())):
    #         yield {'joint': inputs[0], **{key: inputs[i] for i, key in enumerate(unconfounded_datasets.keys())}}

    return 1


a = create_confounded_mnist_dataset(16)
