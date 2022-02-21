import pickle
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from components import Array, KeyArray
from datasets.confounded_mnist import image_gallery
from models import ClassifierFn, SamplingFn
from run_experiment import to_numpy_iterator

# [[image, parent, do_parent, do_noise], do_image]
CompiledMechanismFn = Callable[[Array, Array, Array], Array]
TestResult = Dict[str, Union['TestResult', NDArray]]
Test = Callable[[KeyArray, Array, Dict[str, Array]], Tuple[TestResult, Dict[str, NDArray]]]
DistanceMetric = Callable[[Array, Array], Array]

gallery_fn = partial(image_gallery, num_images_to_display=100, ncols=10)


def l2(x1: Array, x2: Array) -> Array:
    return np.mean(np.power(x1 - x2, 2.), axis=(1, 2, 3))


def image_sequence_plot(image_seq: np.ndarray, n_cases: int = 10, max_cols: int = 10) -> NDArray:
    image_seq = image_seq[:n_cases, :min(image_seq.shape[1], max_cols)]
    image_seq = np.clip(127.5 * image_seq + 127.5, a_min=0, a_max=255) / 255.
    gallery = np.moveaxis(image_seq, 1, 2).reshape((n_cases * image_seq.shape[2],
                                                    image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4]))
    return gallery


def plot_and_save(image: NDArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()


def effectiveness_test(mechanism_fn: CompiledMechanismFn,
                       parent_name: str,
                       sampling_fn: SamplingFn,
                       classifiers: Dict[str, ClassifierFn]) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Dict[str, NDArray]]:
        parent = parents[parent_name]
        do_parent, _ = sampling_fn(rng, (image.shape[0],))
        do_image = mechanism_fn(image, parent, do_parent)
        do_parents = {**parents, parent_name: do_parent}
        output = {}
        for _parent_name, _parent in do_parents.items():
            _, output[_parent_name] = classifiers[_parent_name]((do_image, _parent))
        test_results = jax.tree_map(np.array, output)
        order = np.argsort(np.argmax(np.array(do_parent), axis=-1))

        plots = {'image': gallery_fn(np.array(image)[order]),
                 'do_image': gallery_fn(np.array(do_image)[order]),
                 'do_nothing': gallery_fn(np.array(mechanism_fn(image, parent, parent))[order])}
        return test_results, plots

    return test


def composition_test(mechanism_fn: CompiledMechanismFn,
                     parent_name: str,
                     distance_metric: DistanceMetric = l2,
                     horizon: int = 10) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Dict[str, NDArray]]:
        distance = []
        parent = parents[parent_name]
        do_image = image
        image_sequence = [do_image]
        for j in range(horizon):
            do_image = mechanism_fn(do_image, parent, parent)
            distance.append(distance_metric(image, do_image))
            image_sequence.append(do_image)
        test_results = {f'distance_{(i + 1):d}': np.array(value) for i, value in enumerate(distance)}
        plots = {'composition': image_sequence_plot(np.moveaxis(np.array(image_sequence), 0, 1))}
        return test_results, plots

    return test


def reversibility_test(mechanism_fn: CompiledMechanismFn,
                       parent_name: str,
                       sampling_fn: SamplingFn,
                       distance_metric: DistanceMetric = l2,
                       cycle_length: int = 2,
                       num_cycles: int = 1) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[TestResult, Dict[str, NDArray]]:
        distance = []
        do_parent_cycle, _ = sampling_fn(rng, (cycle_length - 1, image.shape[0]))
        do_parent_cycle = jnp.concatenate((do_parent_cycle, parents[parent_name][jnp.newaxis]))
        do_parent_cycle = jnp.concatenate([do_parent_cycle] * num_cycles, axis=0)
        do_image = image
        image_sequence = [do_image]
        do_parents = parents
        for do_parent in do_parent_cycle:
            do_image = mechanism_fn(do_image, do_parents[parent_name], do_parent)
            distance.append(distance_metric(image, do_image))
            image_sequence.append(do_image)
            do_parents = {**do_parents, parent_name: do_parent}

        output = {f'distance_{(i + 1):d}': np.array(value) for i, value in
                  enumerate(distance[(cycle_length - 1):-1:cycle_length])}
        plots = {'reversibility': image_sequence_plot(np.moveaxis(np.array(image_sequence), 0, 1))}
        return output, plots

    return test


def perform_tests(job_dir: Path,
                  mechanism_fns: Dict[str, CompiledMechanismFn],
                  is_invertible: Dict[str, bool],
                  sampling_fns: Dict[str, SamplingFn],
                  classifiers: Dict[str, ClassifierFn],
                  test_set: Any) -> TestResult:
    plot = True
    parent_names = mechanism_fns.keys()
    tests: Dict[str, Dict[str, Test]] = {parent_name: {} for parent_name in parent_names}
    for parent_name, mechanism_fn in mechanism_fns.items():
        sampling_fn = sampling_fns[parent_name]
        tests[parent_name]['effectiveness'] = effectiveness_test(mechanism_fn, parent_name, sampling_fn, classifiers)
        tests[parent_name]['composition'] = composition_test(mechanism_fn, parent_name)
        if is_invertible[parent_name]:
            tests[parent_name]['reversibility'] = reversibility_test(mechanism_fn, parent_name, sampling_fn,
                                                                     num_cycles=5)

    rng = random.PRNGKey(0)
    test_results: TestResult = {}
    test_set = to_numpy_iterator(test_set, 512, drop_remainder=False)

    for image, parents in test_set:
        rng, _ = jax.random.split(rng)
        batch_results = {key: dict.fromkeys(['effectiveness', 'composition', 'reversibility']) for key in parent_names}
        for parent_name in parent_names:
            for test_name, test_fn in tests[parent_name].items():
                batch_results[parent_name][test_name], plots = test_fn(rng, image, parents)
                if plot:
                    for name, image_plot in plots.items():
                        plot_and_save(image_plot, job_dir / 'plots' / parent_name / f'{name}.png')
        plot = False
        if len(test_results) == 0:
            test_results.update(batch_results)
        else:
            test_results = jax.tree_map(lambda x, y: np.concatenate((x, y)), test_results, batch_results)

    with open(job_dir / 'results.pickle', mode='wb') as f:
        pickle.dump(test_results, f)

    return test_results


def print_nested_dict(nested_dict: Dict, key: Tuple[str, ...] = ()) -> None:
    for sub_key, value in nested_dict.items():
        new_key = key + (str(sub_key),)
        if isinstance(value, dict):
            print_nested_dict(value, new_key)
        else:
            print(new_key, value)


def print_test_results(trees: Iterable[Any]) -> None:
    tree_of_stacks = jax.tree_map(lambda *x: np.stack(x), *trees)

    def print_fn(value: NDArray, precision: str = '.4f') -> str:
        per_seed_mean = np.mean(value, axis=-1)
        return f'{np.mean(per_seed_mean):{precision}} {np.std(per_seed_mean):{precision}}'

    print_nested_dict(jax.tree_map(print_fn, tree_of_stacks))

# def permute_transform_test(mechanism_fns: Dict[str, Mechanism], parent_dims: Dict[str, int], noise_dim: int) -> Test:
#     def test(image: Array, parents: Dict[str, Array]) -> Array:
#
#         do_parents = {p_name: jnp.eye(p_dim)[np.random.randint(0, p_dim, size=image.shape[0])]
#                       for p_name, p_dim in parent_dims.items()}
#         noise = np.zeros(shape=(image.shape[0], noise_dim))
#
#         l2_loss = []
#         images_to_plot = [image, np.zeros_like(image)]
#         images_to_calc = []
#         for parent_order in itertools.permutations(mechanism_fns, len(mechanism_fns)):
#             _image, _parents = image, parents
#             for parent_name in parent_order:
#                 mechanism_fn = mechanism_fns[parent_name]
#                 _image, _ = mechanism_fn(_image, _parents, do_parents[parent_name], noise)
#                 _parents = {**_parents, parent_name: do_parents[parent_name]}
#                 images_to_plot.append(_image)
#             images_to_plot.append(np.zeros_like(image))
#             images_to_calc.append(_image)
#             # images.append(np.zeros_like(_image))
#
#         for im1, im2 in list(itertools.combinations(images_to_calc, 2)):
#             l2_loss.append(l2(im1, im2))
#             images_to_plot.append(np.abs(im1 - im2))
#
#         return np.moveaxis(np.array(l2_loss), 0, 1), np.moveaxis(np.array(images_to_plot), 0, 1)
#
#     return test
