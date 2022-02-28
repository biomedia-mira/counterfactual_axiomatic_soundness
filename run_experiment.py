import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple
from typing import FrozenSet

import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental.optimizers import Optimizer
from numpy.typing import NDArray

from components import Array, InitFn, Params, Shape, StaxLayer
from identifiability_tests import perform_tests, print_test_results
from models import classifier, ClassifierFn, partial_mechanism, MechanismFn, SamplingFn
from models.partial_mechanism import get_sampling_fn
from trainer import train


@dataclass(frozen=True)
class ClassifierConfig:
    parent_name: str
    parent_dims: Dict[str, int]
    input_shape: Shape
    layers: Iterable[StaxLayer]
    # training parameters
    optimizer: Optimizer
    batch_size: int
    num_steps: int
    log_every: int
    eval_every: int
    save_every: int

    @property
    def num_classes(self) -> int:
        return self.parent_dims[self.parent_name]


@dataclass(frozen=True)
class MechanismConfig:
    parent_name: str
    parent_dims: Dict[str, int]
    input_shape: Shape
    critic_layers: Iterable[StaxLayer]
    mechanism: Tuple[InitFn, MechanismFn]
    marginal_dist: NDArray
    is_invertible: bool
    condition_divergence_on_parents: bool
    constraint_function_exponent: int
    # training parameters
    optimizer: Optimizer
    batch_size: int
    num_steps: int
    log_every: int
    eval_every: int
    save_every: int

    @property
    def parent_dim(self) -> int:
        return self.parent_dims[self.parent_name]

    @property
    def sampling_fn(self) -> SamplingFn:
        return get_sampling_fn(self.parent_dim, is_continuous=False, marginal_dist=self.marginal_dist)


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE))


def compile_fn(fn: Callable, params: Params) -> Callable:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return fn(params, *args, **kwargs)

    return _fn


def train_classifier(job_dir: Path,
                     seed: int,
                     train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                     test_dataset: tf.data.Dataset,
                     config: ClassifierConfig) -> ClassifierFn:
    model = classifier(config.num_classes, config.layers, config.optimizer)
    target_dist = frozenset((config.parent_name,))
    select_parent = lambda image, parents: (image, parents[config.parent_name])
    train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent), config.batch_size)
    test_data = to_numpy_iterator(test_dataset.map(select_parent), config.batch_size, drop_remainder=False)
    params = train(model=model,
                   input_shape=config.input_shape,
                   job_dir=job_dir / config.parent_name,
                   num_steps=config.num_steps,
                   seed=seed,
                   train_data=train_data,
                   test_data=test_data,
                   log_every=config.log_every,
                   eval_every=config.eval_every,
                   save_every=config.save_every)
    return compile_fn(fn=model[1], params=params)


def train_mechanism(job_dir: Path,
                    seed: int,
                    classifiers: Dict[str, ClassifierFn],
                    train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                    test_dataset: tf.data.Dataset,
                    config: MechanismConfig) -> Callable[[Array, Array, Array], Array]:
    source_dist: FrozenSet[str] = frozenset()
    target_dist = frozenset((config.parent_name,))

    model = partial_mechanism(source_dist,
                              config.parent_dims,
                              config.parent_name,
                              classifiers,
                              config.critic_layers,
                              config.mechanism,
                              config.sampling_fn,
                              config.is_invertible,
                              config.optimizer,
                              config.condition_divergence_on_parents,
                              config.constraint_function_exponent)

    train_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                                        target_dist: train_datasets[target_dist]}), config.batch_size)
    test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset,
                                                       target_dist: test_dataset}), config.batch_size,
                                  drop_remainder=False)
    params = train(model=model,
                   input_shape=config.input_shape,
                   job_dir=job_dir / f'do_{config.parent_name}',
                   train_data=train_data,
                   test_data=test_data,
                   num_steps=config.num_steps,
                   seed=seed,
                   log_every=1,
                   eval_every=250,
                   save_every=250)
    return compile_fn(fn=config.mechanism[1], params=params[1])


def run_experiment(job_dir: Path,
                   overwrite: bool,
                   seeds: Iterable[int],
                   train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                   test_dataset: tf.data.Dataset,
                   parent_dims: Dict[str, int],
                   classifier_configs: Dict[str, ClassifierConfig],
                   mechanism_configs: Dict[str, MechanismConfig],
                   pseudo_oracles: Dict[str, ClassifierFn]) -> None:
    assert all([parent_name == config.parent_name for parent_name, config in classifier_configs.items()])
    assert all([parent_name == config.parent_name for parent_name, config in mechanism_configs.items()])
    assert all([parent_dims == set(config.parent_dims) for config in classifier_configs.values()])
    assert all([parent_dims == set(config.parent_dims) for config in mechanism_configs.values()])
    has_no_repeats = lambda lst: len(set(lst)) == len(lst)
    assert has_no_repeats([config.parent_name for config in classifier_configs.values()])
    assert has_no_repeats([config.parent_name for config in mechanism_configs.values()])
    assert pseudo_oracles.keys() == parent_dims.keys()

    for seed in seeds:
        seed_dir = job_dir / f'seed_{seed:d}'
        if seed_dir.exists() and (seed_dir / 'results.pickle').exists():
            if overwrite:
                shutil.rmtree(seed_dir)
            else:
                continue

        classifiers = {parent_name: train_classifier(job_dir=seed_dir,
                                                     seed=seed,
                                                     train_datasets=train_datasets,
                                                     test_dataset=test_dataset,
                                                     config=config)
                       for parent_name, config in classifier_configs.items()}

        mechanisms = {parent_name: train_mechanism(job_dir=seed_dir,
                                                   seed=seed,
                                                   classifiers=classifiers,
                                                   train_datasets=train_datasets,
                                                   test_dataset=test_dataset,
                                                   config=config)
                      for parent_name, config in mechanism_configs.items()}

        is_invertible = {parent_name: config.is_invertible for parent_name, config in mechanism_configs.items()}
        sampling_fns = {parent_name: config.sampling_fn for parent_name, config in mechanism_configs.items()}

        test_results = perform_tests(seed_dir, mechanisms, is_invertible, sampling_fns, pseudo_oracles, test_dataset)
        with open(seed_dir / 'results.pickle', mode='wb') as f:
            pickle.dump(test_results, f)

        list_of_test_results = []
        for subdir in job_dir.iterdir():
            with open(subdir / 'results.pickle', mode='rb') as f:
                list_of_test_results.append(pickle.load(f))
        print_test_results(list_of_test_results)
