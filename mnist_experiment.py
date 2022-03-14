import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from jax.experimental import optimizers
from jax.experimental.stax import Conv, ConvTranspose, Dense, Flatten, LeakyRelu, Tanh

from components.stax_extension import PixelNorm2D, ResBlock, Reshape
from datasets.confounded_mnist import digit_colour_scenario, digit_fracture_colour_scenario
from identifiability_tests import perform_tests, print_test_results
from models import ClassifierFn, MechanismFn, classifier, functional_counterfactual, vae_gan
from trainer import train
from utils import compile_fn, prep_classifier_data, prep_mechanism_data

# import matplotlib
# matplotlib.use('tkagg')
tf.config.experimental.set_visible_devices([], 'GPU')

scenarios = {'digit_colour_scenario': digit_colour_scenario,
             'digit_fracture_colour_scenario': digit_fracture_colour_scenario}

layers = \
    (ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(64 * 2, filter_shape=(4, 4), strides=(2, 2)),
     ResBlock(64 * 3, filter_shape=(4, 4), strides=(2, 2)),
     Flatten, Dense(128), LeakyRelu)

mechanism_encoder_layers = \
    (Conv(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     Conv(128, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     Reshape((-1, 7 * 7 * 128)), Dense(1024), LeakyRelu)

# mechanism_decoder_layers = \
#     (Dense(7 * 7 * 128), LeakyRelu, Reshape((-1, 7, 7, 128)),
#      ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
#      Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'), Tanh)


mechanism_decoder_layers = \
    (Dense(7 * 7 * 128), LeakyRelu, Reshape((-1, 7, 7, 128)),
     ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     ConvTranspose(64, filter_shape=(4, 4), strides=(2, 2), padding='SAME'), PixelNorm2D, LeakyRelu,
     Conv(3, filter_shape=(3, 3), strides=(1, 1), padding='SAME'))

base_lr = 1e-4
schedule = optimizers.piecewise_constant(boundaries=[5000, 10000], values=[base_lr, base_lr / 2, base_lr / 8])
mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)
# schedule = optimizers.piecewise_constant(boundaries=[2000, 4000], values=[1e-5, 1e-5 / 2, 1e-5 / 8])
# mechanism_optimizer = optimizers.adam(step_size=schedule, b1=0.0, b2=.9)


def run_experiment(job_dir: Path,
                   data_dir: Path,
                   scenario_name: str,
                   overwrite: bool,
                   seeds: List[int],
                   baseline: bool,
                   partial_mechanisms: bool,
                   confound: bool,
                   de_confound: bool,
                   constraint_function_power: int,
                   from_joint: bool = True) -> None:
    job_name = Path(f'confound_{str(confound)}_de_confounded_{str(de_confound)}_M_{constraint_function_power:d}')
    scenario_dir = job_dir / scenario_name
    pseudo_oracle_dir = scenario_dir / 'pseudo_oracles'
    experiment_dir = job_dir / scenario_dir / job_name

    scenario_fn = scenarios[scenario_name]

    # train pseudo-oracles if not trained
    train_datasets, test_dataset, parent_dims, _, _, input_shape = scenario_fn(data_dir, False, False)
    parent_names = tuple(parent_dims.keys())

    pseudo_oracles: Dict[str, ClassifierFn] = {}
    for parent_name, parent_dim in parent_dims.items():
        model = classifier(num_classes=parent_dims[parent_name], layers=layers)
        train_data, test_data = prep_classifier_data(parent_name, train_datasets, test_dataset, batch_size=1024)
        params = train(model=model,
                       job_dir=pseudo_oracle_dir / parent_name,
                       seed=100,
                       train_data=train_data,
                       test_data=test_data,
                       input_shape=input_shape,
                       optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                       num_steps=20,  # 2000,
                       log_every=1,
                       eval_every=50,
                       save_every=50)
        pseudo_oracles[parent_name] = compile_fn(fn=model[1], params=params)

    # run experiment
    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape \
        = scenario_fn(data_dir, confound, de_confound)

    for seed in seeds:
        seed_dir = experiment_dir / f'seed_{seed:d}'
        if seed_dir.exists() and (seed_dir / 'results.pickle').exists():
            if overwrite:
                shutil.rmtree(seed_dir)
            else:
                pass

        # train classifiers
        classifiers: Dict[str, ClassifierFn] = {}
        for parent_name, parent_dim in parent_dims.items():
            model = classifier(num_classes=parent_dims[parent_name], layers=layers)
            train_data, test_data = prep_classifier_data(parent_name, train_datasets, test_dataset, batch_size=1024)
            params = train(model=model,
                           job_dir=seed_dir / parent_name,
                           seed=seed,
                           train_data=train_data,
                           test_data=test_data,
                           input_shape=input_shape,
                           optimizer=optimizers.adam(step_size=5e-4, b1=0.9),
                           num_steps=20,  # 2000,
                           log_every=1,
                           eval_every=50,
                           save_every=50)
            classifiers[parent_name] = compile_fn(fn=model[1], params=params)

        # train (partial) mechanisms
        mechanisms: Dict[str, MechanismFn] = {}
        for parent_name in (parent_names if partial_mechanisms and not baseline else ['all']):
            if baseline:
                model, get_mechanism_fn = vae_gan(parent_dims=parent_dims,
                                                  latent_dim=64,
                                                  critic_layers=layers,
                                                  encoder_layers=mechanism_encoder_layers,
                                                  decoder_layers=mechanism_decoder_layers,
                                                  condition_divergence_on_parents=True,
                                                  from_joint=from_joint)
            else:
                model, get_mechanism_fn = functional_counterfactual(do_parent_name=parent_name,
                                                                    parent_dims=parent_dims,
                                                                    classifiers=classifiers,
                                                                    critic_layers=layers,
                                                                    marginal_dists=marginals,
                                                                    mechanism_encoder_layers=mechanism_encoder_layers,
                                                                    mechanism_decoder_layers=mechanism_decoder_layers,
                                                                    is_invertible=is_invertible,
                                                                    condition_divergence_on_parents=True,
                                                                    constraint_function_power=constraint_function_power,
                                                                    from_joint=from_joint)
            train_data, test_data = prep_mechanism_data(parent_name, parent_names, from_joint, train_datasets,
                                                        test_dataset, batch_size=512)
            params = train(model=model,
                           job_dir=seed_dir / f'do_{parent_name}',
                           seed=seed,
                           train_data=train_data,
                           test_data=test_data,
                           input_shape=input_shape,
                           optimizer=mechanism_optimizer,
                           num_steps=20000,
                           log_every=1,
                           eval_every=250,
                           save_every=250)
            mechanisms[parent_name] = get_mechanism_fn(params[1])

        mechanisms = dict.fromkeys(parent_names, mechanisms['all']) if not partial_mechanisms else mechanisms

        test_results = perform_tests(seed_dir, mechanisms, is_invertible, marginals, pseudo_oracles, test_dataset)
        with open(seed_dir / 'results.pickle', mode='wb') as f:
            pickle.dump(test_results, f)

    list_of_test_results = []
    for subdir in experiment_dir.iterdir():
        with open(subdir / 'results.pickle', mode='rb') as f:
            list_of_test_results.append(pickle.load(f))
    print_test_results(list_of_test_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', dest='job_dir', type=Path, help='job-dir where logs and models are saved')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, help='data-dir where files will be saved')
    parser.add_argument('--scenario-name', dest='scenario_name', type=str, help='Name of scenario to run.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite an existing run')
    parser.add_argument('--seeds', dest='seeds', nargs="+", type=int, help='list of random seeds')
    args = parser.parse_args()

    run_experiment(args.job_dir, args.data_dir, args.scenario_name, args.overwrite, args.seeds,
                   partial_mechanisms=True, baseline=True, confound=True, de_confound=True, constraint_function_power=1)

# for confound, de_confound, constraint_function_exponent in product(*parameter_space.values()):
#     if not confound and de_confound:
#         continue
# parameter_space = {'confound': [True, False], 'de_confound': [True, False], 'constraint_function_exponent': [1, 3]}
