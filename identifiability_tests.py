import itertools
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from numpy.typing import NDArray
from tqdm import tqdm

from datasets.utils import PMF, ParentDist, Scenario
from models.utils import AuxiliaryFn, CouterfactualFn
from staxplus import Array, KeyArray
from utils import flatten_nested_dict

TestResult = Union[Dict[str, Array], Dict[str, 'TestResult']]
Plots = Union[NDArray[np.uint8], Dict[str, NDArray[np.uint8]]]
Test = Callable[[KeyArray, Array, Dict[str, Array]], Tuple[TestResult, Plots]]


def decode_fn(x: Array) -> Array:
    return jnp.clip(127.5 * x + 127.5, a_min=0, a_max=255)


# Calculates the average pixel l1 distance in the range of 0-255
def l1(x1: Array, x2: Array) -> Array:
    return jnp.mean(jnp.abs(decode_fn(x1) - decode_fn(x2)), axis=(1, 2, 3))


def sequence_plot(image_seq: NDArray[Any],
                  n_cases: int = 10,
                  max_cols: int = 10) -> NDArray[np.uint8]:
    image_seq = image_seq[:n_cases, :min(image_seq.shape[1], max_cols)]
    new_shape = (n_cases * image_seq.shape[2], image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4])
    gallery = np.moveaxis(image_seq, 1, 2).reshape(new_shape)
    return gallery


class PlotMode(Enum):
    DEFAULT = 1
    LOW_DENSITY = 2
    HIGH_DENSITY = 3


def pseudo_oracle_test(pseudo_oracles: Dict[str, AuxiliaryFn]) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:
        test_result = {}
        for parent_name, parent in parents.items():
            _, test_result[parent_name] = pseudo_oracles[parent_name](image=image, parent=parent)
        return test_result, {}
    return test


def effectiveness_test(counterfactual_fn: CouterfactualFn,
                       parent_dist: ParentDist,
                       pseudo_oracles: Dict[str, AuxiliaryFn],
                       joint_pmf: Optional[PMF],
                       _decode_fn: Callable[[Array], Array] = decode_fn,
                       plot_cases_per_row: int = 3,
                       sep_width: int = 1) -> Test:
    parent_name = parent_dist.name

    def _plot(image: Array,
              do_nothing: Array,
              do_image: Array,
              parents: Dict[str, Array],
              do_parents: Dict[str, Array],
              mode: PlotMode):
        nrows = 10 if not parent_dist.is_discrete else parent_dist.dim
        ncols = 3 * plot_cases_per_row
        height, width, channels = image.shape[1:]
        dim = parent_dist.dim
        if joint_pmf is not None:
            _, binned_parents, _ = joint_pmf(parents)
            prob, binned_do_parents, _ = joint_pmf(do_parents)
            parent = binned_parents[parent_name]
            do_parent = binned_do_parents[parent_name]
            ind = [jnp.where(jnp.logical_and(jnp.not_equal(parent, do_parent), do_parent == i))[0] for i in range(dim)]
            if mode == PlotMode.LOW_DENSITY:
                ind = [_ind[jnp.argsort(jnp.take(prob, _ind))] for _ind in ind]
            elif mode == PlotMode.HIGH_DENSITY:
                ind = [_ind[jnp.argsort(-jnp.take(prob, _ind))] for _ind in ind]
        else:
            do_parent = jnp.argmax(do_parents[parent_name], axis=-1)
            parent = jnp.argmax(parents[parent_name], axis=-1)
            if parent_dist.is_discrete:
                ind = [jnp.where(jnp.logical_and(jnp.not_equal(parent, do_parent), do_parent == i))[0]
                       for i in range(dim)]
            else:
                ind = jnp.argsort(do_parent, axis=0)

        indices = jnp.concatenate([ind[i][:plot_cases_per_row * (nrows // dim)] for i in range(dim)])
        im = _decode_fn(jnp.stack((image, do_nothing, do_image), axis=1))
        im = np.array(jnp.reshape(im[indices], (-1, *image.shape[1:]))).astype(np.uint8)
        _im = np.zeros((nrows*ncols, height, width, channels), dtype=np.uint8)
        _im[:len(im)] = im
        im = _im
        plot = im.reshape((nrows, ncols, height, width, channels)).swapaxes(1, 2).reshape(height * nrows,
                                                                                          width * ncols, channels)
        if sep_width > 0:
            for i in range(3, ncols, 3):
                start, stop = width * i - sep_width // 2, width * i + sep_width // 2 + sep_width % 2
                plot[:, start:stop, :] = 255

        return plot

    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:
        do_parent = parent_dist.sample(rng, (image.shape[0],))
        do_parents = {**parents, parent_name: do_parent}
        do_image = counterfactual_fn(rng, image, parents, do_parents)
        test_result = {}
        for _parent_name, _parent in do_parents.items():
            _, test_result[_parent_name] = pseudo_oracles[_parent_name](image=do_image, parent=_parent)
            test_result[_parent_name]['target'] = _parent
        do_nothing = counterfactual_fn(rng, image, parents, parents)
        plots = {'default': _plot(image, do_nothing, do_image, parents, do_parents, PlotMode.DEFAULT),
                 'low_density': _plot(image, do_nothing, do_image, parents, do_parents, PlotMode.LOW_DENSITY),
                 'high_density': _plot(image, do_nothing, do_image, parents, do_parents, PlotMode.HIGH_DENSITY)}
        return test_result, plots
    return test


def composition_test(counterfactual_fn: CouterfactualFn,
                     horizon: int = 10) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:
        image_sequence = [image]
        do_image = image
        for _ in range(horizon):
            do_image = counterfactual_fn(rng, do_image, parents, parents)
            image_sequence.append(do_image)
        test_result = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        image_sequece = np.array(decode_fn(jnp.moveaxis(jnp.array(image_sequence), 0, 1))).astype(np.uint8)
        plot = sequence_plot(image_sequece, max_cols=9)
        return test_result, plot

    return test


def reversibility_test(counterfactual_fn: CouterfactualFn,
                       parent_dist: ParentDist,
                       cycle_length: int = 2,
                       num_cycles: int = 1) -> Test:
    parent_name = parent_dist.name

    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:
        do_parent_cycle = parent_dist.sample(rng, (cycle_length - 1, image.shape[0]))
        do_parent_cycle = jnp.concatenate((do_parent_cycle, parents[parent_name][jnp.newaxis]))
        do_parent_cycle = jnp.concatenate([do_parent_cycle] * num_cycles, axis=0)
        image_sequence = [image]
        do_image = image
        for do_parent in list(do_parent_cycle):
            do_parents = {**parents, parent_name: do_parent}
            do_image = counterfactual_fn(rng, do_image, parents, do_parents)
            image_sequence.append(do_image)
            parents = do_parents
        output = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        image_sequece = np.array(decode_fn(jnp.moveaxis(jnp.array(image_sequence), 0, 1))).astype(np.uint8)
        plot = sequence_plot(image_sequece, max_cols=9)
        return output, plot

    return test


def commutativity_test(counterfactual_fn: Dict[str, CouterfactualFn],
                       parent_dist_1: ParentDist,
                       parent_dist_2: ParentDist,
                       sep_width: int = 1, ) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:

        image_sequence = []
        for parent_order in itertools.permutations((parent_dist_1, parent_dist_2)):
            _image, _parents = image, parents
            image_sequence.append(image)
            for parent_dist in parent_order:
                mechanism_fn = counterfactual_fn[parent_dist.name]
                _do_parents = {**_parents, parent_dist.name: parent_dist.sample(rng, (image.shape[0],))}
                _image = mechanism_fn(rng, _image, _parents, _do_parents)
                _parents = _do_parents
                image_sequence.append(_image)
        im1, im2 = image_sequence[2], image_sequence[-1]
        image_sequence.append(im1 - im2 - 1.)
        output = {'distance': l1(im1, im2)}
        width = image.shape[2]
        image_sequece = np.array(decode_fn(jnp.moveaxis(jnp.array(image_sequence), 0, 1))).astype(np.uint8)
        plot = sequence_plot(image_sequece, max_cols=9)
        if sep_width > 0:
            for i in [3, 6]:
                start, stop = width * i - sep_width // 2, width * i + sep_width // 2 + sep_width % 2
                plot[:, start:stop, :] = 255

        return output, plot

    return test


def format_results(trees: Iterable[Any], summarise: bool = True, print_results: bool = True) -> Dict[str, Any]:
    tree_of_stacks = tree_map(lambda *x: jnp.stack(x), *trees)

    def print_fn(value: Array, precision: str = '.2f') -> str:
        per_seed_mean = jnp.mean(value, axis=-1)
        return f'{jnp.mean(per_seed_mean):{precision}} ({jnp.std(per_seed_mean):{precision}})'

    def formatter(orignal_dict: Dict[Any, Any], new_dict: Dict[Any, Any]) -> Dict[Any, Any]:
        for key, value in orignal_dict.items():
            if isinstance(value, dict):
                new_dict[key] = {}
                new_dict[key] = formatter(orignal_dict[key], new_dict[key])
            else:
                if summarise and key not in (['accuracy', 'absolute_error', 'distance_1', 'distance_2', 'distance_10']):
                    continue
                if key in ['accuracy', 'absolute_error']:
                    value = value * 100
                new_dict[key] = print_fn(value)
        return new_dict
    formatted_results = formatter(tree_of_stacks, {})

    if print_results:
        print(yaml.dump(formatted_results, default_flow_style=False))
    return formatted_results


def get_tests(parent_dists: Dict[str, ParentDist],
              joint_pmf: PMF,
              pseudo_oracles: Dict[str, AuxiliaryFn],
              counterfactual_fns: Dict[str, CouterfactualFn]) -> Dict[str, Union[Test, Dict[str, Test]]]:
    tests = {}
    tests['pseudo_oracle_quality'] = pseudo_oracle_test(pseudo_oracles)
    for parent_name, parent_dist in parent_dists.items():
        if parent_name not in counterfactual_fns:
            continue
        tests[parent_name] = {}
        mechanism_fn = counterfactual_fns[parent_name]
        tests[parent_name]['effectiveness'] = effectiveness_test(mechanism_fn, parent_dist, pseudo_oracles, joint_pmf)
        tests[parent_name]['composition'] = composition_test(mechanism_fn)
        tests[parent_name]['reversibility'] = reversibility_test(mechanism_fn, parent_dist, num_cycles=5)
    for p1, p2 in itertools.combinations(parent_dists.values(), 2):
        if p1.name not in counterfactual_fns or p2.name not in counterfactual_fns:
            continue
        tests[f'{p1.name}_{p2.name}_commutativity'] = commutativity_test(counterfactual_fns, p1, p2)
    return tests


def run_tests_on_data(tests: Dict[str, Union[Test, Dict[str, Test]]],
                      test_set: tf.data.Dataset,
                      num_batches_to_plot: int = 1) -> Tuple[TestResult, Dict[str, NDArray[Any]]]:
    rng = random.PRNGKey(0)
    results: TestResult = {}
    figures: Dict[str, NDArray[Any]] = {}
    plot_counter = 0
    for image, parents in tqdm(tfds.as_numpy(test_set)):
        rng, _ = jax.random.split(rng)
        res = tree_map(lambda func: func(rng, image, parents), tests, is_leaf=lambda leaf: callable(leaf))
        flat, treedef = tree_flatten(res, is_leaf=lambda x: isinstance(x, tuple))
        output, plots = [tree_unflatten(treedef, [el[i] for el in flat]) for i in (0, 1)]
        results = output if not results else tree_map(lambda x, y: jnp.concatenate((x, y)), results, output)
        if plot_counter < num_batches_to_plot:
            for key, fig in flatten_nested_dict(plots).items():
                fig = fig if fig.shape[-1] == 3 else np.repeat(fig, repeats=3, axis=-1)
                figures[('_'.join(key) + f'_{plot_counter:d}.png')] = fig
            plot_counter += 1
    return results, figures


def evaluate(result_dir: Path,
             scenario: Scenario,
             test_set: tf.data.Dataset,
             counterfactual_fns: Dict[str, CouterfactualFn],
             pseudo_oracles: Dict[str, AuxiliaryFn],
             num_batches_to_plot: int = 1,
             overwrite: bool = False) -> TestResult:
    results_path = (result_dir / 'results.pickle')
    if results_path.exists() and not overwrite:
        with open(results_path, mode='rb') as f:
            return pickle.load(f)
    parent_dists, joint_pmf = scenario.parent_dists, scenario.joint_pmf
    assert pseudo_oracles.keys() == parent_dists.keys()
    tests = get_tests(parent_dists, joint_pmf, pseudo_oracles, counterfactual_fns)

    results, figures = run_tests_on_data(tests, test_set, num_batches_to_plot)
    for figure_name, figure in figures.items():
        path = result_dir / Path('plots') / Path(figure_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(path), figure)

    with open(result_dir / 'results.pickle', mode='wb') as f:
        pickle.dump(results, f)
    return results
