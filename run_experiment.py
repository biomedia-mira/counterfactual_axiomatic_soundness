from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental.optimizers import Optimizer
from jax.experimental.stax import Dense, Flatten, serial

from components import Array, InitFn, KeyArray, Params, Shape, StaxLayer
from components.f_gan import f_gan
from models import classifier, ClassifierFn, functional_counterfactual, MechanismFn, SamplingFn
from trainer import train


def broadcast(array: Array, shape: Tuple[int, ...]) -> Array:
    return jnp.broadcast_to(jnp.expand_dims(array, axis=tuple(range(1, 1 + len(shape) - array.ndim))), shape)


def condition_on_parents(parent_dims: Dict[str, int]) -> StaxLayer:
    def init_fn(rng: KeyArray, shape: Shape) -> Tuple[Shape, Params]:
        return (*shape[:-1], shape[-1] + sum(parent_dims.values())), ()

    def apply_fn(params: Params, inputs: Any, **kwargs: Any) -> Array:
        image, parents = inputs
        shape = (*image.shape[:-1], sum(parent_dims.values()))
        _parents = jnp.concatenate([parents[key] for key in parent_dims.keys()], axis=-1)
        return jnp.concatenate((image, broadcast(_parents, shape)), axis=-1)

    return init_fn, apply_fn


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))


def compile_fn(fn: Callable, params: Params) -> Callable:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return fn(params, *args, **kwargs)

    return _fn


def train_classifier(job_dir: Path,
                     seed: int,
                     parent_name: str,
                     num_classes: int,
                     layers: Iterable[StaxLayer],
                     train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                     test_dataset: tf.data.Dataset,
                     input_shape: Shape,
                     optimizer: Optimizer,
                     batch_size: int,
                     num_steps: int) -> ClassifierFn:
    model = classifier(num_classes, layers, optimizer)
    model_path = job_dir / parent_name / 'model.npy'
    if model_path.exists():
        params = np.load(str(model_path), allow_pickle=True)
    else:
        target_dist = frozenset((parent_name,))
        select_parent = lambda image, parents: (image, parents[parent_name])
        train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent), batch_size)
        test_data = to_numpy_iterator(test_dataset.map(select_parent), batch_size)
        params = train(model=model,
                       input_shape=input_shape,
                       job_dir=job_dir / parent_name,
                       num_steps=num_steps,
                       seed=seed,
                       train_data=train_data,
                       test_data=test_data,
                       log_every=1,
                       eval_every=50,
                       save_every=50)
    return compile_fn(fn=model[1], params=params)


def train_mechanism(job_dir: Path,
                    seed: int,
                    parent_name: str,
                    parent_dims: Dict[str, int],
                    classifiers: Dict[str, ClassifierFn],
                    critic_layers: Iterable[StaxLayer],
                    mechanism: Tuple[InitFn, MechanismFn],
                    sampling_fn: SamplingFn,
                    is_invertible: bool,
                    train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                    test_dataset: tf.data.Dataset,
                    input_shape: Shape,
                    optimizer: Optimizer,
                    batch_size: int,
                    num_steps: int) -> Callable[[Array, Array, Array], Array]:
    source_dist: FrozenSet[str] = frozenset()
    target_dist = frozenset((parent_name,))
    critic = serial(condition_on_parents(parent_dims), *critic_layers, Flatten, Dense(1))
    divergence = f_gan(critic, mode='gan', trick_g=True)
    model = functional_counterfactual(source_dist,
                                      parent_name,
                                      classifiers,
                                      divergence,
                                      mechanism,
                                      sampling_fn,
                                      is_invertible,
                                      optimizer)

    model_path = job_dir / f'do_{parent_name}' / 'model.npy'
    if model_path.exists():
        params = np.load(str(model_path), allow_pickle=True)
    else:
        train_data = to_numpy_iterator(
            tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                 target_dist: train_datasets[target_dist]}), batch_size)
        test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset,
                                                           target_dist: test_dataset}), batch_size)
        params = train(model=model,
                       input_shape=input_shape,
                       job_dir=job_dir / f'do_{parent_name}',
                       train_data=train_data,
                       test_data=test_data,
                       num_steps=num_steps,
                       seed=seed,
                       log_every=1,
                       eval_every=250,
                       save_every=250)
    # return compile_fn(fn=critic[1], params=params[0])
    return compile_fn(fn=mechanism[1], params=params[1])
