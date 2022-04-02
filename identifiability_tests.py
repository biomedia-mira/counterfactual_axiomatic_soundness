import itertools
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from numpy.typing import NDArray

from components import Array, KeyArray
from models import ClassifierFn, MarginalDistribution, MechanismFn
from utils import flatten_nested_dict, to_numpy_iterator

TestResult = Dict[str, Union['TestResult', NDArray]]
Test = Callable[[KeyArray, Array, Dict[str, Array]], Tuple[TestResult, NDArray]]


def decode_fn(x: NDArray) -> NDArray:
    return np.clip(127.5 * x + 127.5, a_min=0, a_max=255).astype(int)


# Calculates the average pixel l1 distance in the range of 0-255
def l1(x1: Array, x2: Array) -> Array:
    return np.mean(np.abs(decode_fn(np.array(x1)) - decode_fn(np.array(x2))), axis=(1, 2, 3))


def sequence_plot(image_seq: NDArray,
                  n_cases: int = 10,
                  max_cols: int = 10,
                  _decode_fn: Callable[[NDArray], NDArray] = decode_fn) -> NDArray:
    image_seq = image_seq[:n_cases, :min(image_seq.shape[1], max_cols)]
    image_seq = _decode_fn(image_seq)
    gallery = np.moveaxis(image_seq, 1, 2).reshape((n_cases * image_seq.shape[2],
                                                    image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4]))
    return gallery


def plot_and_save(image: NDArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)
    plt.close()


def effectiveness_test(mechanism_fn: MechanismFn,
                       parent_name: str,
                       marginal: MarginalDistribution,
                       pseudo_oracles: Dict[str, ClassifierFn],
                       _decode_fn: Callable[[NDArray], NDArray] = decode_fn,
                       plot_cases_per_row: int = 3,
                       sep_width: int = 1) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, NDArray]:
        do_parent = marginal.sample(rng, (image.shape[0],))
        do_parents = {**parents, parent_name: do_parent}
        do_image = mechanism_fn(rng, image, parents, do_parents)
        output = {}
        for _parent_name, _parent in do_parents.items():
            _, output[_parent_name] = pseudo_oracles[_parent_name]((do_image, _parent))
        test_results = jax.tree_map(np.array, output)
        do_nothing = mechanism_fn(rng, image, parents, parents)

        # plot
        nrows, ncols = marginal.marginal_dist.shape[0], 3 * plot_cases_per_row
        height, width, channels = image.shape[1:]
        im = _decode_fn(np.stack((image, do_nothing, do_image), axis=1))
        _parents, _do_parents = np.argmax(parents[parent_name], axis=-1), np.argmax(do_parents[parent_name], axis=-1)
        indices = np.concatenate(
            [np.where(np.logical_and(np.not_equal(_parents, _do_parents), _do_parents == i))[0][:plot_cases_per_row]
             for i in range(nrows)])
        im = np.reshape(im[indices], (-1, *image.shape[1:]))
        plot = im.reshape((nrows, ncols, height, width, channels)).swapaxes(1, 2).reshape(height * nrows,
                                                                                          width * ncols, channels)
        if sep_width > 0:
            for i in range(3, ncols, 3):
                start, stop = width * i - sep_width // 2, width * i + sep_width // 2 + sep_width % 2
                plot[:, start:stop, :] = 255

        return test_results, plot

    return test


def composition_test(mechanism_fn: MechanismFn,
                     horizon: int = 10) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, NDArray]:
        image_sequence = [image]
        do_image = image
        for j in range(horizon):
            do_image = mechanism_fn(rng, do_image, parents, parents)
            image_sequence.append(do_image)
        test_results = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        plot = sequence_plot(np.moveaxis(np.array(image_sequence), 0, 1), max_cols=9)
        return test_results, plot

    return test


def reversibility_test(mechanism_fn: MechanismFn,
                       parent_name: str,
                       marginal: MarginalDistribution,
                       cycle_length: int = 2,
                       num_cycles: int = 1) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, NDArray]:
        do_parent_cycle = marginal.sample(rng, (cycle_length - 1, image.shape[0]))
        do_parent_cycle = jnp.concatenate((do_parent_cycle, parents[parent_name][jnp.newaxis]))
        do_parent_cycle = jnp.concatenate([do_parent_cycle] * num_cycles, axis=0)
        image_sequence = [image]
        do_image = image
        for do_parent in do_parent_cycle:
            do_parents = {**parents, parent_name: do_parent}
            do_image = mechanism_fn(rng, do_image, parents, do_parents)
            image_sequence.append(do_image)
            parents = do_parents
        output = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        plot = sequence_plot(np.moveaxis(np.array(image_sequence), 0, 1), max_cols=9)
        return output, plot

    return test


def commutativity_test(mechanism_fns: Dict[str, MechanismFn],
                       marginals: Dict[str, MarginalDistribution],
                       parent_name_1: str,
                       parent_name_2: str,
                       sep_width: int = 1, ) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, NDArray]:

        image_sequence = []
        for parent_order in itertools.permutations((parent_name_1, parent_name_2)):
            _image, _parents = image, parents
            image_sequence.append(image)
            for parent_name in parent_order:
                mechanism_fn = mechanism_fns[parent_name]
                marginal = marginals[parent_name]
                _do_parents = {**_parents, parent_name: marginal.sample(rng, (image.shape[0],))}
                _image = mechanism_fn(rng, _image, _parents, _do_parents)
                _do_parents = _parents
                image_sequence.append(_image)
        im1, im2 = image_sequence[2], image_sequence[-2]
        image_sequence.append(im1 - im2 - 1.)
        output = {'distance': l1(im1, im2)}
        # plot
        width = image.shape[2]
        plot = sequence_plot(np.moveaxis(np.array(image_sequence), 0, 1), max_cols=9)
        if sep_width > 0:
            for i in [3, 6]:
                start, stop = width * i - sep_width // 2, width * i + sep_width // 2 + sep_width % 2
                plot[:, start:stop, :] = 255

        return output, plot

    return test


def print_test_results(trees: Iterable[Any]) -> None:
    tree_of_stacks = jax.tree_map(lambda *x: np.stack(x), *trees)

    def print_fn(value: NDArray, precision: str = '.4f') -> str:
        per_seed_mean = np.mean(value, axis=-1)
        return f'{np.mean(per_seed_mean):{precision}} {np.std(per_seed_mean):{precision}}'

    for key, value in flatten_nested_dict(jax.tree_map(print_fn, tree_of_stacks)).items():
        print(key, value)


def evaluate(job_dir: Path,
             mechanism_fns: Dict[str, MechanismFn],
             is_invertible: Dict[str, bool],
             marginals: Dict[str, MarginalDistribution],
             pseudo_oracles: Dict[str, ClassifierFn],
             test_set: Any,
             num_batches_to_plot: int = 1,
             overwrite: bool = False) -> TestResult:
    results_path = (job_dir / 'results.pickle')
    if results_path.exists() and not overwrite:
        with open(results_path, mode='rb') as f:
            return pickle.load(f)

    assert pseudo_oracles.keys() == is_invertible.keys() == marginals.keys()
    parent_names = marginals.keys()
    tests = {}
    for parent_name, marginal in marginals.items():
        tests[parent_name] = {}
        mechanism_fn = mechanism_fns[parent_name]
        tests[parent_name]['effectiveness'] = effectiveness_test(mechanism_fn, parent_name, marginal, pseudo_oracles)
        tests[parent_name]['composition'] = composition_test(mechanism_fn)
        if is_invertible[parent_name]:
            tests[parent_name]['reversibility'] = reversibility_test(mechanism_fn, parent_name, marginal, num_cycles=5)
    for p1, p2 in itertools.combinations(parent_names, 2):
        tests[f'{p1}_{p2}_commutativity'] = commutativity_test(mechanism_fns, marginals, p1, p2)

    rng = random.PRNGKey(0)
    results: TestResult = {}
    test_set = to_numpy_iterator(test_set, 512, drop_remainder=False)
    plot_counter = 0
    for image, parents in test_set:
        rng, _ = jax.random.split(rng)
        res = jax.tree_map(lambda func: func(rng, image, parents), tests, is_leaf=lambda leaf: callable(leaf))
        flat, treedef = tree_flatten(res, is_leaf=lambda x: isinstance(x, tuple))
        output, plots = [tree_unflatten(treedef, [el[i] for el in flat]) for i in (0, 1)]
        results = output if not results else jax.tree_map(lambda x, y: np.concatenate((x, y)), results, output)
        if plot_counter < num_batches_to_plot:
            for key, value in flatten_nested_dict(plots).items():
                plot_and_save(value, job_dir / 'plots' / ('_'.join(key) + f'_{plot_counter:d}.png'))
            plot_counter += 1

    with open(job_dir / 'results.pickle', mode='wb') as f:
        pickle.dump(results, f)

    return results
