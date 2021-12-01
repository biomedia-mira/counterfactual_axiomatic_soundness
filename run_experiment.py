import shutil
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental.optimizers import Optimizer
from jax.experimental.stax import Dense, Flatten, serial
from numpy.typing import NDArray

from components import Array, InitFn, KeyArray, Params, Shape, StaxLayer
from models import MechanismFn, classifier, functional_counterfactual
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


def run_experiment(job_dir: Path,
                   seed: int,
                   train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                   test_dataset: tf.data.Dataset,
                   parent_dims: Dict[str, int],
                   marginals: Dict[str, NDArray],
                   input_shape: Shape,
                   # Classifier
                   classifier_layers: Iterable[StaxLayer],
                   classifier_optimizer: Optimizer,
                   classifier_batch_size: int,
                   classifier_num_steps: int,
                   # ConfoundingFn
                   critic_layers: Iterable[StaxLayer],
                   abductor_layers: Iterable[StaxLayer],
                   mechanism_constructor: Callable[[int, int], Tuple[InitFn, MechanismFn]],
                   noise_dim: int,
                   counterfactual_batch_size: int,
                   counterfactual_num_steps: int,
                   # Misc
                   overwrite: bool = False
                   ):
    job_dir = Path(job_dir)
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)
    # Train classifiers
    classifiers = {}
    for parent_name, parent_dim in parent_dims.items():
        model = classifier(parent_dim, classifier_layers, classifier_optimizer)
        model_path = job_dir / parent_name / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            target_dist = frozenset((parent_name,))
            select_parent = lambda image, parents: (image, parents[parent_name])
            train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent), classifier_batch_size)
            test_data = to_numpy_iterator(test_dataset.map(select_parent), classifier_batch_size)
            params = train(model=model,
                           input_shape=input_shape,
                           job_dir=job_dir / parent_name,
                           num_steps=classifier_num_steps,
                           seed=seed,
                           train_data=train_data,
                           test_data=test_data,
                           log_every=1,
                           eval_every=50,
                           save_every=50)
        classifiers[parent_name] = compile_fn(fn=model[1], params=params)

    mechanisms_compiled = {}
    for parent_name, parent_dim in parent_dims.items():
        source_dist, target_dist = frozenset(), frozenset((parent_name,))
        critic = serial(condition_on_parents(parent_dims), *critic_layers, Flatten, Dense(1))
        abductor = serial(condition_on_parents(parent_dims), *abductor_layers, Flatten, Dense(noise_dim))
        mechanism = mechanism_constructor(parent_dim, noise_dim)
        model = functional_counterfactual(source_dist, parent_name, marginals[parent_name],
                                          classifiers, critic, mechanism, abductor)
        model_path = job_dir / f'do_{parent_name}' / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            train_data = to_numpy_iterator(
                tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                     target_dist: train_datasets[target_dist]}), counterfactual_batch_size)
            test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset,
                                                               target_dist: test_dataset}), counterfactual_batch_size)
            params = train(model=model,
                           input_shape=input_shape,
                           job_dir=job_dir / f'do_{parent_name}',
                           train_data=train_data,
                           test_data=test_data,
                           num_steps=counterfactual_num_steps,
                           seed=seed,
                           log_every=1,
                           eval_every=250,
                           save_every=250)
        mechanisms_compiled[parent_name] = compile_fn(fn=mechanism[1], params=params[1])

    # Test
    # repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
    #                for p_name, mechanism in mechanisms.items()}
    # cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
    #               for p_name, mechanism in mechanisms.items()}
    # permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
    #                                                    for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
    # tests = {**repeat_test, **cycle_test, **permute_test}
    # res = perform_tests(test_data, tests)
