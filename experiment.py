from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Sequence, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.utils import Array, Scenario
from models.conditional_vae import conditional_vae
from models.auxiliary_model import auxiliary_model
from models.functional_counterfactual import functional_counterfactual
from models.last_try import functional_counterfactual_vae
from models.utils import AuxiliaryFn, MechanismFn
from staxplus import GradientTransformation, StaxLayer
from staxplus.train import train


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size,
                                    drop_remainder=drop_remainder,
                                    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))


def prep_mechanism_data(do_parent_name: str,
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
                         overwrite: bool) -> Dict[str, AuxiliaryFn]:
    train_datasets, test_dataset, parent_dists, input_shape, _ = scenario
    discriminative_models: Dict[str, AuxiliaryFn] = {}
    for parent_name, parent_dist in parent_dists.items():
        model, get_discriminative_fn = auxiliary_model(parent_dist, backbone=backbone)
        target_dist = frozenset((parent_name,))

        def select_parent(image: Array, parents: Dict[str, Array]) -> Tuple[Array, Array]:
            return image, parents[parent_name]
        train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent),
                                       batch_size=train_config.batch_size)
        test_data = to_numpy_iterator(test_dataset.map(select_parent), batch_size=train_config.batch_size,
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
        discriminative_models[parent_name] = get_discriminative_fn(params)
    return discriminative_models


def get_baseline(job_dir: Path,
                 seed: int,
                 scenario: Scenario,
                 pseudo_oracles: Dict[str, AuxiliaryFn],
                 vae_encoder: StaxLayer,
                 vae_decoder: StaxLayer,
                 aux_backbone: StaxLayer,
                 aux_train_config: TrainConfig,
                 train_config: TrainConfig,
                 from_joint: bool,
                 overwrite: bool) -> Dict[str, MechanismFn]:
    train_datasets, test_dataset, parent_dists, input_shape, _ = scenario
    parent_names = list(parent_dists.keys())
    parent_name = 'all'
    aux_models = get_auxiliary_models(job_dir=job_dir,
                                      seed=seed,
                                      scenario=scenario,
                                      backbone=aux_backbone,
                                      train_config=aux_train_config,
                                      overwrite=overwrite)
    model, get_mechanism_fn = conditional_vae(parent_dists=parent_dists,
                                              oracles=aux_models,
                                              aux_models=aux_models,
                                              vae_encoder=vae_encoder,
                                              vae_decoder=vae_decoder,
                                              from_joint=from_joint)
    # model, get_mechanism_fn = functional_counterfactual_vae(do_parent_name=parent_name,
    #                                                         parent_dists=parent_dists,
    #                                                         vae_encoder=vae_encoder,
    #                                                         vae_decoder=vae_decoder,
    #                                                         oracles=pseudo_oracles,
    #                                                         aux_backbone=aux_backbone,
    #                                                         from_joint=from_joint)

    train_data, test_data = prep_mechanism_data(parent_name, parent_names, from_joint, train_datasets,
                                                test_dataset, train_config.batch_size)
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
    mechanisms = {parent_name: get_mechanism_fn(params) for parent_name in parent_names}
    return mechanisms


def get_mechanisms(job_dir: Path,
                   seed: int,
                   scenario: Scenario,
                   partial_mechanisms: bool,
                   constraint_function_power: int,
                   pseudo_oracles: Dict[str, AuxiliaryFn],
                   critic: StaxLayer,
                   mechanism: StaxLayer,
                   train_config: TrainConfig,
                   from_joint: bool,
                   overwrite: bool) -> Dict[str, MechanismFn]:
    train_datasets, test_dataset, parent_dists, input_shape, _ = scenario
    parent_names = list(parent_dists.keys())
    mechanisms: Dict[str, MechanismFn] = {}
    for parent_name in (parent_names if partial_mechanisms else ['all']):
        model, get_mechanism_fn = functional_counterfactual(do_parent_name=parent_name,
                                                            parent_dists=parent_dists,
                                                            oracles=pseudo_oracles,
                                                            critic=critic,
                                                            mechanism=mechanism,
                                                            constraint_function_power=constraint_function_power,
                                                            from_joint=from_joint)
        train_data, test_data = prep_mechanism_data(parent_name, parent_names, from_joint, train_datasets,
                                                    test_dataset, batch_size=train_config.batch_size)
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
        mechanisms[parent_name] = get_mechanism_fn(params)
    mechanisms = {parent_name: mechanisms['all'] for parent_name in parent_names} if 'all' in mechanisms else mechanisms
    return mechanisms
