import shutil
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.typing import NDArray

from components import Params, Shape, StaxLayer
from models import Abductor, Critic, Mechanism, classifier, functional_counterfactual
from trainer import train


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE))


def compile_fn(fn: Callable, params: Params) -> Callable:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        return fn(params, *args, **kwargs)

    return _fn


def run_experiment(job_dir: Path,
                   # Data
                   train_datasets: Dict[FrozenSet[str], tf.data.Dataset],
                   test_dataset: tf.data.Dataset,
                   parent_dims: Dict[str, int],
                   marginals: Dict[str, NDArray],
                   input_shape: Shape,
                   # Classifier
                   classifier_layers: Iterable[StaxLayer],
                   classifier_batch_size: int,
                   classifier_num_steps: int,
                   # Mechanism
                   interventions: Optional[Iterable[Tuple[str, ...]]],
                   critic: Critic,
                   abductor: Abductor,
                   mechanisms: Dict[str, Mechanism],
                   mechanism_batch_size: int,
                   mechanism_num_steps: int,
                   # Misc
                   seed: int = 1,
                   overwrite: bool = False) -> Dict[str, Callable]:
    job_dir = Path(job_dir)
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)

    # Train classifiers
    classifiers = {}
    for parent_name, parent_dim in parent_dims.items():
        classifier_model = classifier(parent_dim, classifier_layers)
        model_path = job_dir / parent_name / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            target_dist = frozenset((parent_name,))
            select_parent = lambda image, parents: (image, parents[parent_name])
            train_data = to_numpy_iterator(train_datasets[target_dist].map(select_parent), classifier_batch_size)
            test_data = to_numpy_iterator(test_dataset.map(select_parent), classifier_batch_size)
            params = train(model=classifier_model,
                           input_shape=input_shape,
                           job_dir=job_dir / parent_name,
                           num_steps=classifier_num_steps,
                           seed=seed,
                           train_data=train_data,
                           test_data=test_data,
                           log_every=1,
                           eval_every=50,
                           save_every=50)
        classifiers[parent_name] = compile_fn(fn=classifier_model[1], params=params)

    # Train mechanisms
    mechanisms_compiled = {}
    for intervention in interventions:
        parent_name = intervention[0]
        source_dist = frozenset()
        target_dist = frozenset(intervention)
        model = functional_counterfactual(source_dist, parent_name, marginals[parent_name], classifiers, critic,
                                          mechanisms[parent_name], abductor)
        model_path = job_dir / f'do_{parent_name}' / 'model.npy'
        if model_path.exists():
            params = np.load(str(model_path), allow_pickle=True)
        else:
            train_data = to_numpy_iterator(
                tf.data.Dataset.zip({source_dist: train_datasets[source_dist],
                                     target_dist: train_datasets[target_dist]}),
                mechanism_batch_size)
            test_data = to_numpy_iterator(tf.data.Dataset.zip({source_dist: test_dataset,
                                                               target_dist: test_dataset}),
                                          mechanism_batch_size)
            params = train(model=model,
                           input_shape=input_shape,
                           job_dir=job_dir / f'do_{parent_name}',
                           train_data=train_data,
                           test_data=test_data,
                           num_steps=mechanism_num_steps,  # 5000
                           seed=seed,
                           log_every=1,
                           eval_every=250,
                           save_every=250)
        mechanisms_compiled[parent_name] = compile_fn(mechanisms[parent_name][1], params[1])
    return mechanisms_compiled
    # Test
    # repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
    #                for p_name, mechanism in mechanisms.items()}
    # cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
    #               for p_name, mechanism in mechanisms.items()}
    # permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
    #                                                    for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
    # tests = {**repeat_test, **cycle_test, **permute_test}
    # res = perform_tests(test_data, tests)
