from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from jax.example_libraries.optimizers import Optimizer

from components.stax_extension import StaxLayer
from datasets.confounded_mnist import Scenario
from models import classifier, ClassifierFn, conditional_vae, functional_counterfactual, MechanismFn
from train import train
from utils import compile_fn, prep_classifier_data, prep_mechanism_data


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    optimizer: Optimizer
    num_steps: int
    log_every: int
    eval_every: int
    save_every: int


def get_classifiers(job_dir: Path,
                    seed: int,
                    scenario: Scenario,
                    classifier_layers: Sequence[StaxLayer],
                    train_config: TrainConfig,
                    overwrite: bool) -> Dict[str, ClassifierFn]:
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    classifiers: Dict[str, ClassifierFn] = {}
    for parent_name, parent_dim in parent_dims.items():
        model = classifier(num_classes=parent_dims[parent_name], layers=classifier_layers)
        train_data, test_data = prep_classifier_data(parent_name, train_datasets, test_dataset, train_config.batch_size)
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
        classifiers[parent_name] = compile_fn(fn=model[1], params=params)
    return classifiers


def get_baseline(job_dir: Path,
                 seed: int,
                 scenario: Scenario,
                 vae_encoder: StaxLayer,
                 vae_decoder: StaxLayer,
                 train_config: TrainConfig,
                 from_joint: bool,
                 overwrite: bool) -> Dict[str, MechanismFn]:
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    parent_names = list(parent_dims.keys())
    parent_name = 'all'
    model, get_mechanism_fn = conditional_vae(parent_dims=parent_dims,
                                              marginal_dists=marginals,
                                              vae_encoder=vae_encoder,
                                              vae_decoder=vae_decoder,
                                              from_joint=from_joint)
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
                   classifier_layers: Sequence[StaxLayer],
                   classifier_train_config: TrainConfig,
                   critic: StaxLayer,
                   mechanism: StaxLayer,
                   train_config: TrainConfig,
                   from_joint: bool,
                   overwrite: bool) -> Dict[str, MechanismFn]:
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario
    parent_names = list(parent_dims.keys())
    classifiers \
        = get_classifiers(job_dir / 'classifiers', seed, scenario, classifier_layers, classifier_train_config,
                          overwrite)
    mechanisms: Dict[str, MechanismFn] = {}
    for parent_name in (parent_names if partial_mechanisms else ['all']):
        model, get_mechanism_fn = functional_counterfactual(do_parent_name=parent_name,
                                                            parent_dims=parent_dims,
                                                            marginal_dists=marginals,
                                                            classifiers=classifiers,
                                                            critic=critic,
                                                            mechanism=mechanism,
                                                            is_invertible=is_invertible,
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
