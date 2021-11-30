import shutil
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental import optimizers
from numpy.typing import NDArray

from components.classifier import classifier
from components.functional_counterfactual import model_wrapper
from components.stax_extension import Params, Shape, StaxLayer, StaxLayerConstructor
from trainer import train


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))


def compile_fn(fn: Callable, params: Params) -> Callable:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return fn(params, *args, **kwargs)

    return _fn


def run_experiment(job_dir: Path,
                   train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                   test_dataset: tf.data.Dataset,
                   parent_dims: Dict[str, int],
                   marginals: Dict[str, NDArray],
                   input_shape: Shape,
                   classifier_layers: Iterable[StaxLayer],
                   mechanism_constructor: StaxLayerConstructor,
                   critic_constructor: StaxLayerConstructor,
                   from_joint: bool = True,
                   overwrite: bool = False) -> None:
    seed = 100
    job_dir = Path(job_dir)
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)

    # Train classifiers
    classifiers = {}
    batch_size = 1024
    for parent_name, parent_dim in parent_dims.items():
        classifier_model = classifier(parent_dim, classifier_layers, optimizers.adam(step_size=5e-4, b1=0.9))
        model_path = job_dir / parent_name / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            select_parent = lambda image, parents: (image, parents[parent_name])
            target_dist = frozenset((parent_name,))
            train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent), batch_size)
            test_data = to_numpy_iterator(test_dataset.map(select_parent), batch_size)
            params = train(model=classifier_model,
                           input_shape=input_shape,
                           job_dir=job_dir / parent_name,
                           num_steps=2000,
                           seed=seed,
                           train_data=train_data,
                           test_data=test_data,
                           log_every=1,
                           eval_every=100,
                           save_every=100)
        classifiers[parent_name] = compile_fn(fn=classifier_model[1], params=params)

    # Train counterfactual functions
    interventions = (('digit',), ('color',))
    batch_size = 512
    parent_names = parent_dims.keys()
    for intervention in interventions:
        source_dist = frozenset() if from_joint else frozenset(parent_names)
        target_dist = frozenset(intervention) if from_joint else frozenset(parent_names)
        train_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                                            target_dist: train_datasets[target_dist]}), batch_size)
        test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset,
                                                           target_dist: test_dataset}), batch_size)

        parent_name = intervention[0]

        model, _ = model_wrapper(source_dist, parent_name, marginals[parent_name], classifiers, critic, mechanism,
                                 abductor)

        params = train(model=model,
                       input_shape=input_shape,
                       job_dir=job_dir,
                       num_steps=5000,
                       train_data=train_data,
                       test_data=test_data,
                       log_every=1,
                       eval_every=500,
                       save_every=500)

    #
    # classifiers, divergences, mechanisms = build_functions(params, *functions)
    # repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
    #                for p_name, mechanism in mechanisms.items()}
    # cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
    #               for p_name, mechanism in mechanisms.items()}
    # permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
    #                                                    for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
    # tests = {**repeat_test, **cycle_test, **permute_test}
    # res = perform_tests(test_data, tests)
