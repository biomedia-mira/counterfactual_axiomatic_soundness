from typing import Any
from typing import Callable
from typing import Dict, FrozenSet, Iterable, Sequence, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from components import Params


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE))


def compile_fn(fn: Callable, params: Params) -> Callable:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return fn(params, *args, **kwargs)

    return _fn


def prep_classifier_data(parent_name: str,
                         train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                         test_dataset: tf.data.Dataset,
                         batch_size: int) -> Tuple[Iterable, Iterable]:
    target_dist = frozenset((parent_name,))
    select_parent = lambda image, parents: (image, parents[parent_name])
    train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent), batch_size=batch_size)
    test_data = to_numpy_iterator(test_dataset.map(select_parent), batch_size=batch_size, drop_remainder=False)
    return train_data, test_data


def prep_mechanism_data(do_parent_name: str,
                        parent_names: Sequence[str],
                        from_joint: bool,
                        train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                        test_dataset: tf.data.Dataset,
                        batch_size: int) -> Tuple[Iterable, Iterable]:
    do_parent_names = tuple(parent_names) if do_parent_name == 'all' else (do_parent_name,)
    source_dist = frozenset() if from_joint else frozenset(parent_names)
    target_dist = frozenset(do_parent_names) if from_joint else frozenset(parent_names)
    train_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                                        target_dist: train_datasets[target_dist]}),
                                   batch_size=batch_size)
    test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset, target_dist: test_dataset}),
                                  batch_size=batch_size, drop_remainder=False)
    return train_data, test_data
