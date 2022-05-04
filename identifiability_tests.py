from enum import Enum
import itertools
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from numpy.typing import NDArray
from tqdm import tqdm

from datasets.utils import ParentDist, Scenario, PMF
from experiment import to_numpy_iterator
from models.utils import AuxiliaryFn, MechanismFn
from staxplus import Array, KeyArray
from utils import flatten_nested_dict
import yaml


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


def effectiveness_test(mechanism_fn: MechanismFn,
                       parent_dist: ParentDist,
                       pseudo_oracles: Dict[str, AuxiliaryFn],
                       joint_pmf: PMF,
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
        nrows, ncols = 10, 3 * plot_cases_per_row
        height, width, channels = image.shape[1:]

        _, binned_parents, _ = joint_pmf(parents)
        prob, binned_do_parents, dims = joint_pmf(do_parents)
        parent = binned_parents[parent_name]
        do_parent = binned_do_parents[parent_name]
        dim = dims[parent_name]

        ind = [jnp.where(jnp.logical_and(jnp.not_equal(parent, do_parent), do_parent == i))[0] for i in range(dim)]
        if mode == PlotMode.LOW_DENSITY:
            ind = [_ind[jnp.argsort(jnp.take(prob, _ind))] for _ind in ind]
        elif mode == PlotMode.HIGH_DENSITY:
            ind = [_ind[jnp.argsort(-jnp.take(prob, _ind))] for _ind in ind]

        indices = jnp.concatenate([ind[i][:plot_cases_per_row * (nrows // dim)] for i in range(dim)])
        im = _decode_fn(jnp.stack((image, do_nothing, do_image), axis=1))
        im = np.array(jnp.reshape(im[indices], (-1, *image.shape[1:]))).astype(np.uint8)
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
        do_image = mechanism_fn(rng, image, parents, do_parents)
        test_result = {}
        for _parent_name, _parent in do_parents.items():
            _, test_result[_parent_name] = pseudo_oracles[_parent_name](image=do_image, parent=_parent)
            test_result[_parent_name]['target'] = _parent
        do_nothing = mechanism_fn(rng, image, parents, parents)
        plots = {'default': _plot(image, do_nothing, do_image, parents, do_parents, PlotMode.DEFAULT),
                 'low_density': _plot(image, do_nothing, do_image, parents, do_parents, PlotMode.LOW_DENSITY),
                 'high_density': _plot(image, do_nothing, do_image, parents, do_parents, PlotMode.HIGH_DENSITY)}
        return test_result, plots
    return test


def composition_test(mechanism_fn: MechanismFn,
                     horizon: int = 10) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:
        image_sequence = [image]
        do_image = image
        for _ in range(horizon):
            do_image = mechanism_fn(rng, do_image, parents, parents)
            image_sequence.append(do_image)
        test_result = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        image_sequece = np.array(decode_fn(jnp.moveaxis(jnp.array(image_sequence), 0, 1))).astype(np.uint8)
        plot = sequence_plot(image_sequece, max_cols=9)
        return test_result, plot

    return test


def reversibility_test(mechanism_fn: MechanismFn,
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
            do_image = mechanism_fn(rng, do_image, parents, do_parents)
            image_sequence.append(do_image)
            parents = do_parents
        output = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        image_sequece = np.array(decode_fn(jnp.moveaxis(jnp.array(image_sequence), 0, 1))).astype(np.uint8)
        plot = sequence_plot(image_sequece, max_cols=9)
        return output, plot

    return test


def commutativity_test(mechanism_fns: Dict[str, MechanismFn],
                       parent_dist_1: ParentDist,
                       parent_dist_2: ParentDist,
                       sep_width: int = 1, ) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Plots]:

        image_sequence = []
        for parent_order in itertools.permutations((parent_dist_1, parent_dist_2)):
            _image, _parents = image, parents
            image_sequence.append(image)
            for parent_dist in parent_order:
                mechanism_fn = mechanism_fns[parent_dist.name]
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


def print_test_results(trees: Iterable[Any]) -> None:
    tree_of_stacks = tree_map(lambda *x: jnp.stack(x), *trees)

    def print_fn(value: Array, precision: str = '.4f') -> str:
        per_seed_mean = jnp.mean(value, axis=-1)
        return f'{jnp.mean(per_seed_mean):{precision}} {jnp.std(per_seed_mean):{precision}}'
    print(yaml.dump(tree_map(print_fn, tree_of_stacks), default_flow_style=False))


def evaluate(job_dir: Path,
             scenario: Scenario,
             mechanism_fns: Dict[str, MechanismFn],
             pseudo_oracles: Dict[str, AuxiliaryFn],
             num_batches_to_plot: int = 1,
             overwrite: bool = False) -> TestResult:
    results_path = (job_dir / 'results.pickle')
    if results_path.exists() and not overwrite:
        with open(results_path, mode='rb') as f:
            return pickle.load(f)
    parent_dists, joint_pmf = scenario.parent_dists, scenario.joint_pmf
    assert pseudo_oracles.keys() == parent_dists.keys()
    tests = {}
    for parent_name, parent_dist in parent_dists.items():
        tests[parent_name] = {}
        mechanism_fn = mechanism_fns[parent_name]
        tests[parent_name]['effectiveness'] = effectiveness_test(mechanism_fn, parent_dist, pseudo_oracles, joint_pmf)
        tests[parent_name]['composition'] = composition_test(mechanism_fn)
        if parent_dist.is_invertible:
            tests[parent_name]['reversibility'] = reversibility_test(mechanism_fn, parent_dist, num_cycles=5)
    for p1, p2 in itertools.combinations(parent_dists.values(), 2):
        tests[f'{p1.name}_{p2.name}_commutativity'] = commutativity_test(mechanism_fns, p1, p2)

    rng = random.PRNGKey(0)
    results: TestResult = {}
    test_set = to_numpy_iterator(scenario.test_data, 512, drop_remainder=False)
    plot_counter = 0
    for image, parents in tqdm(test_set):
        rng, _ = jax.random.split(rng)
        res = tree_map(lambda func: func(rng, image, parents), tests, is_leaf=lambda leaf: callable(leaf))
        flat, treedef = tree_flatten(res, is_leaf=lambda x: isinstance(x, tuple))
        output, plots = [tree_unflatten(treedef, [el[i] for el in flat]) for i in (0, 1)]
        results = output if not results else tree_map(lambda x, y: jnp.concatenate((x, y)), results, output)
        if plot_counter < num_batches_to_plot:
            for key, image in flatten_nested_dict(plots).items():
                path = job_dir / 'plots' / ('_'.join(key) + f'_{plot_counter:d}.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(str(path), image)

            plot_counter += 1

    with open(job_dir / 'results.pickle', mode='wb') as f:
        pickle.dump(results, f)

    return results
