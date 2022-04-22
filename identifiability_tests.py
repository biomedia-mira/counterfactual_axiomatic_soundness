import itertools
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import tensorflow as tf
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

from experiment import to_numpy_iterator
from models.utils import DiscriminativeFn, MechanismFn, ParentDist
from staxplus import Array, KeyArray
from utils import flatten_nested_dict

TestResult = Union[Dict[str, Array], Dict[str, 'TestResult']]
Test = Callable[[KeyArray, Array, Dict[str, Array]], Tuple[TestResult, Array]]


def decode_fn(x: Array) -> Array:
    return jnp.clip(127.5 * x + 127.5, a_min=0, a_max=255).astype(int)


# Calculates the average pixel l1 distance in the range of 0-255
def l1(x1: Array, x2: Array) -> Array:
    return jnp.mean(jnp.abs(decode_fn(x1) - decode_fn(x2)), axis=(1, 2, 3))


def sequence_plot(image_seq: Array,
                  n_cases: int = 10,
                  max_cols: int = 10,
                  _decode_fn: Callable[[Array], Array] = decode_fn) -> Array:
    image_seq = image_seq[:n_cases, :min(image_seq.shape[1], max_cols)]
    image_seq = _decode_fn(image_seq)
    gallery = jnp.moveaxis(image_seq, 1, 2).reshape((n_cases * image_seq.shape[2],
                                                    image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4]))
    return gallery


def plot_and_save(image: Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    # plt.show(block=False)
    plt.close()


def effectiveness_test(mechanism_fn: MechanismFn,
                       parent_dist: ParentDist,
                       pseudo_oracles: Dict[str, DiscriminativeFn],
                       _decode_fn: Callable[[Array], Array] = decode_fn,
                       plot_cases_per_row: int = 3,
                       sep_width: int = 1) -> Test:
    parent_name = parent_dist.name

    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Array]:
        do_parent = parent_dist.sample(rng, (image.shape[0],))
        do_parents = {**parents, parent_name: do_parent}
        do_image = mechanism_fn(rng, image, parents, do_parents)
        test_result = {}
        for _parent_name, _parent in do_parents.items():
            _, test_result[_parent_name] = pseudo_oracles[_parent_name](image=do_image, parent=_parent)
        do_nothing = mechanism_fn(rng, image, parents, parents)

        # plot
        nrows, ncols = 10, 3 * plot_cases_per_row
        height, width, channels = image.shape[1:]
        im = _decode_fn(jnp.stack((image, do_nothing, do_image), axis=1))
        _parents, _do_parents = jnp.argmax(parents[parent_name], axis=-1), jnp.argmax(do_parents[parent_name], axis=-1)
        indices = jnp.concatenate(
            [jnp.where(jnp.logical_and(jnp.not_equal(_parents, _do_parents), _do_parents == i))[0][:plot_cases_per_row]
             for i in range(nrows)])
        im = jnp.reshape(im[indices], (-1, *image.shape[1:]))
        plot = im.reshape((nrows, ncols, height, width, channels)).swapaxes(1, 2).reshape(height * nrows,
                                                                                          width * ncols, channels)
        if sep_width > 0:
            for i in range(3, ncols, 3):
                start, stop = width * i - sep_width // 2, width * i + sep_width // 2 + sep_width % 2
                plot[:, start:stop, :] = 255

        return test_result, plot

    return test


def composition_test(mechanism_fn: MechanismFn,
                     horizon: int = 10) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Array]:
        image_sequence = [image]
        do_image = image
        for _ in range(horizon):
            do_image = mechanism_fn(rng, do_image, parents, parents)
            image_sequence.append(do_image)
        test_result = {f'distance_{i:d}': l1(image, _do_image) for i, _do_image in enumerate(image_sequence)}
        plot = sequence_plot(jnp.moveaxis(jnp.array(image_sequence), 0, 1), max_cols=9)
        return test_result, plot

    return test


def reversibility_test(mechanism_fn: MechanismFn,
                       parent_dist: ParentDist,
                       cycle_length: int = 2,
                       num_cycles: int = 1) -> Test:
    parent_name = parent_dist.name

    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Array]:
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
        plot = sequence_plot(jnp.moveaxis(jnp.array(image_sequence), 0, 1), max_cols=9)
        return output, plot

    return test


def commutativity_test(mechanism_fns: Dict[str, MechanismFn],
                       parent_dist_1: ParentDist,
                       parent_dist_2: ParentDist,
                       sep_width: int = 1, ) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Array]:

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
        # plot
        width = image.shape[2]
        plot = sequence_plot(jnp.moveaxis(jnp.array(image_sequence), 0, 1), max_cols=9)
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

    for key, value in flatten_nested_dict(tree_map(print_fn, tree_of_stacks)).items():
        print(key, value)


def evaluate(job_dir: Path,
             parent_dists: Dict[str, ParentDist],
             mechanism_fns: Dict[str, MechanismFn],
             pseudo_oracles: Dict[str, DiscriminativeFn],
             test_set: tf.data.Dataset,
             num_batches_to_plot: int = 1,
             overwrite: bool = False) -> TestResult:
    results_path = (job_dir / 'results.pickle')
    if results_path.exists() and not overwrite:
        with open(results_path, mode='rb') as f:
            return pickle.load(f)

    assert pseudo_oracles.keys() == parent_dists.keys()
    tests = {}
    for parent_name, parent_dist in parent_dists.items():
        tests[parent_name] = {}
        mechanism_fn = mechanism_fns[parent_name]
        tests[parent_name]['effectiveness'] = effectiveness_test(mechanism_fn, parent_dist, pseudo_oracles)
        tests[parent_name]['composition'] = composition_test(mechanism_fn)
        if parent_dist.is_invertible:
            tests[parent_name]['reversibility'] = reversibility_test(mechanism_fn, parent_dist, num_cycles=5)
    for p1, p2 in itertools.combinations(parent_dists.values(), 2):
        tests[f'{p1}_{p2}_commutativity'] = commutativity_test(mechanism_fns, p1, p2)

    rng = random.PRNGKey(0)
    results: TestResult = {}
    test_set = to_numpy_iterator(test_set, 512, drop_remainder=False)
    plot_counter = 0
    for image, parents in test_set:
        rng, _ = jax.random.split(rng)
        res = tree_map(lambda func: func(rng, image, parents), tests, is_leaf=lambda leaf: callable(leaf))
        flat, treedef = tree_flatten(res, is_leaf=lambda x: isinstance(x, tuple))
        output, plots = [tree_unflatten(treedef, [el[i] for el in flat]) for i in (0, 1)]
        results = output if not results else tree_map(lambda x, y: jnp.concatenate((x, y)), results, output)
        if plot_counter < num_batches_to_plot:
            for key, value in flatten_nested_dict(plots).items():
                plot_and_save(value, job_dir / 'plots' / ('_'.join(key) + f'_{plot_counter:d}.png'))
            plot_counter += 1

    with open(job_dir / 'results.pickle', mode='wb') as f:
        pickle.dump(results, f)

    return results
