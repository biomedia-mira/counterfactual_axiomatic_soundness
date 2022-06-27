from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, Sequence, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from typing_extensions import Protocol

from datasets.utils import Array, ParentDist, Scenario
from models.auxiliary_model import auxiliary_model
from models.utils import AuxiliaryFn, CouterfactualFn
from staxplus import GradientTransformation, Model, Params, StaxLayer
from staxplus.train import train


class GetModelFn(Protocol):
    def __call__(self,
                 do_parent_name: str,
                 parent_dists: Dict[str, ParentDist],
                 pseudo_oracles: Dict[str, AuxiliaryFn]) -> Tuple[Model, Callable[[Params], CouterfactualFn]]:
        ...


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size,
                                    drop_remainder=drop_remainder,
                                    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    optimizer: GradientTransformation
    num_steps: int
    log_every: int
    eval_every: int
    save_every: int


def get_auxiliary_models(job_dir: Path,
                         seed: int,
                         scenario: Scenario,
                         backbone: StaxLayer,
                         train_config: TrainConfig,
                         overwrite: bool,
                         simulated_intervervention: bool = True) -> Dict[str, AuxiliaryFn]:
    train_datasets, test_dataset_uc, _, parent_dists, input_shape, _ = scenario
    aux_models: Dict[str, AuxiliaryFn] = {}
    for parent_name, parent_dist in parent_dists.items():
        model, get_aux_fn = auxiliary_model(parent_dist, backbone=backbone)
        target_dist = frozenset((parent_name,)) if simulated_intervervention else frozenset()

        def select_parent(image: Array, parents: Dict[str, Array]) -> Tuple[Array, Array]:
            return image, parents[parent_name]
        train_data = to_numpy_iterator(data=train_datasets[target_dist].map(select_parent),
                                       batch_size=train_config.batch_size)
        test_data = to_numpy_iterator(data=test_dataset_uc.map(select_parent),
                                      batch_size=train_config.batch_size,
                                      drop_remainder=True)
        params = train(model=model,
                       job_dir=job_dir / parent_name,
                       seed=seed,
                       train_data=train_data,
                       test_data=test_data,
                       input_shape=input_shape,
                       optimizer=train_config.optimizer,
                       num_steps=train_config.num_steps,
                       log_every=train_config.log_every,
                       eval_every=train_config.eval_every,
                       save_every=train_config.save_every,
                       overwrite=overwrite)
        aux_models[parent_name] = get_aux_fn(params)
    return aux_models


def _prep_data(do_parent_name: str,
               parent_names: Sequence[str],
               from_joint: bool,
               train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
               test_dataset: tf.data.Dataset,
               batch_size: int) -> Tuple[Iterable[Any], Iterable[Any]]:
    do_parent_names = tuple(parent_names) if do_parent_name == 'all' else (do_parent_name,)
    source_dist = frozenset() if from_joint else frozenset(parent_names)
    target_dist = frozenset(do_parent_names) if from_joint else frozenset(parent_names)
    train_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                                        target_dist: train_datasets[target_dist]}),
                                   batch_size=batch_size)
    test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset, target_dist: test_dataset}),
                                  batch_size=batch_size, drop_remainder=True)
    return train_data, test_data


def get_counterfactual_fns(job_dir: Path,
                           seed: int,
                           scenario: Scenario,
                           get_model_fn: GetModelFn,
                           use_partial_fns: bool,
                           pseudo_oracles: Dict[str, AuxiliaryFn],
                           train_config: TrainConfig,
                           from_joint: bool,
                           overwrite: bool) -> Dict[str, CouterfactualFn]:
    train_datasets, test_dataset_uc, _, parent_dists, input_shape, _ = scenario
    parent_names = list(parent_dists.keys())
    counterfactual_fns: Dict[str, CouterfactualFn] = {}
    for parent_name in (parent_names if use_partial_fns else ['all']):
        model, get_mechanism_fn = get_model_fn(do_parent_name=parent_name,
                                               parent_dists=parent_dists,
                                               pseudo_oracles=pseudo_oracles)

        train_data, test_data = _prep_data(parent_name, parent_names, from_joint, train_datasets,
                                           test_dataset_uc, batch_size=train_config.batch_size)
        params = train(model=model,
                       job_dir=job_dir / f'do_{parent_name}',
                       seed=seed,
                       train_data=train_data,
                       test_data=test_data,
                       input_shape=input_shape,
                       optimizer=train_config.optimizer,
                       num_steps=train_config.num_steps,
                       log_every=train_config.log_every,
                       eval_every=train_config.eval_every,
                       save_every=train_config.save_every,
                       overwrite=overwrite)
        counterfactual_fns[parent_name] = get_mechanism_fn(params)
    counterfactual_fns = {parent_name: counterfactual_fns['all']
                          for parent_name in parent_names} if 'all' in counterfactual_fns else counterfactual_fns
    return counterfactual_fns
