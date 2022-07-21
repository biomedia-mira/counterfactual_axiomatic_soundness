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
                 do_parent_names: Sequence[str],
                 parent_dists: Dict[str, ParentDist],
                 pseudo_oracles: Dict[str, AuxiliaryFn]) -> Tuple[Model, Callable[[Params], CouterfactualFn]]:
        ...


@dataclass(frozen=True)
class TrainConfig:
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
        params = train(model=model,
                       job_dir=job_dir / parent_name,
                       seed=seed,
                       train_data=tfds.as_numpy(train_datasets[target_dist].map(select_parent)),
                       test_data=tfds.as_numpy(test_dataset_uc.map(select_parent)),
                       input_shape=input_shape,
                       optimizer=train_config.optimizer,
                       num_steps=train_config.num_steps,
                       log_every=train_config.log_every,
                       eval_every=train_config.eval_every,
                       save_every=train_config.save_every,
                       overwrite=overwrite)
        aux_models[parent_name] = get_aux_fn(params)
    return aux_models


def _prep_data(do_parent_names: Sequence[str],
               parent_names: Sequence[str],
               simulated_intervention: bool,
               train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
               test_dataset: tf.data.Dataset) -> Tuple[Iterable[Any], Iterable[Any]]:
    assert all([parent_name in parent_names for parent_name in do_parent_names])
    source_dist = frozenset(parent_names) if simulated_intervention else frozenset()
    target_dist = frozenset(parent_names) if simulated_intervention else frozenset(do_parent_names)
    train_data = tfds.as_numpy(tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                                    target_dist: train_datasets[target_dist]}))
    test_data = tfds.as_numpy(tf.data.Dataset.zip({source_dist: test_dataset, target_dist: test_dataset}))
    return train_data, test_data


def get_counterfactual_fns(job_dir: Path,
                           seed: int,
                           scenario: Scenario,
                           parent_names: Sequence[str],
                           get_model_fn: GetModelFn,
                           pseudo_oracles: Dict[str, AuxiliaryFn],
                           train_config: TrainConfig,
                           use_partial_fns: bool,
                           simulated_intervention: bool,
                           overwrite: bool) -> Dict[str, CouterfactualFn]:
    train_datasets, test_dataset_uc, _, parent_dists, input_shape, _ = scenario
    counterfactual_fns: Dict[str, CouterfactualFn] = {}

    for parent_name in (parent_names if use_partial_fns else ('all', )):
        do_parent_names = list(parent_names) if parent_name == 'all' else (parent_name, )
        train_data, test_data = _prep_data(do_parent_names, parent_names, simulated_intervention, train_datasets,
                                           test_dataset_uc)
        model, get_counterfactual_fn = get_model_fn(do_parent_names, parent_dists, pseudo_oracles)
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
        counterfactual_fns[parent_name] = get_counterfactual_fn(params)
    if not use_partial_fns:
        counterfactual_fns = {p_name: counterfactual_fns['all'] for p_name in parent_names}
    return counterfactual_fns
