import itertools
from typing import Any, Callable, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from components import Array, KeyArray
from models import ClassifierFn, SamplingFn

Test = Callable[[KeyArray, Array, Dict[str, Array]], Tuple[Dict[str, NDArray], Optional[NDArray]]]
DistanceMetric = Callable[[Array, Array], Array]
# [[params, [image, parents], [image, parents]], [div_loss, output]]
DivergenceFn = Callable[[Tuple[Array, Array], Tuple[Array, Array]], Tuple[Array, Any]]
# [[params, image, parent, do_parent, do_noise], do_image]
MechanismFn = Callable[[Array, Array, Array], Array]


def l2(x1: Array, x2: Array) -> Array:
    return np.mean(np.power(x1 - x2, 2.), axis=(1, 2, 3))


def effectiveness_test(mechanism_fn: MechanismFn,
                       parent_name: str,
                       sampling_fn: SamplingFn,
                       classifiers: Dict[str, ClassifierFn]) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[Dict[str, NDArray], Optional[NDArray]]:
        parent = parents[parent_name]
        do_parent, _ = sampling_fn(rng, (image.shape[0],))
        do_image = mechanism_fn(image, parent, do_parent)
        do_parents = {**parents, parent_name: do_parent}
        output = {}
        for _parent_name, _parent in do_parents.items():
            _, output[_parent_name] = classifiers[_parent_name]((do_image, _parent))

        return jax.tree_map(np.array, output), None

    return test


def composition_test(mechanism_fn: MechanismFn,
                     parent_name: str,
                     distance_metric: DistanceMetric = l2,
                     horizon: int = 10) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[Dict[str, NDArray], Optional[NDArray]]:
        distance = []
        parent = parents[parent_name]
        do_image = image
        image_sequence = [do_image]
        for j in range(horizon):
            do_image = mechanism_fn(do_image, parent, parent)
            distance.append(distance_metric(image, do_image))
            image_sequence.append(do_image)
        output = {f'distance_{i:d}': np.array(value) for i, value in enumerate(distance)}
        return output, np.moveaxis(np.array(image_sequence), 0, 1)

    return test


def reversibility_test(mechanism_fn: MechanismFn,
                       parent_name: str,
                       sampling_fn: SamplingFn,
                       distance_metric: DistanceMetric = l2,
                       cycle_length: int = 2,
                       num_cycles: int = 1) -> Test:
    def test(rng: KeyArray, image: Array, parents: Dict[str, Array]) -> Tuple[Dict[str, NDArray], Optional[NDArray]]:
        distance = []
        # do_parent_cycle = jnp.eye(parent_dim)[np.random.randint(0, parent_dim, size=(cycle_length - 1, image.shape[0]))]
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

        output = {f'distance_step_{i // cycle_length:d}_cycle_{i % cycle_length:d}': np.array(value)
                  for i, value in enumerate(distance)}

        return output, np.moveaxis(np.array(image_sequence), 0, 1)

    return test


def plot_image_sequence(image_seq: np.ndarray, title: str = '', n_cases: int = 10) -> None:
    image_seq = image_seq[:n_cases]
    gallery = np.moveaxis(image_seq, 1, 2).reshape((n_cases * image_seq.shape[2],
                                                    image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4]))
    plt.imshow(gallery)
    plt.title(title)
    plt.show()


def perform_tests(mechanism_fns: Dict[str, MechanismFn],
                  is_invertible: Dict[str, bool],
                  sampling_fns: Dict[str, SamplingFn],
                  classifiers: Dict[str, ClassifierFn],
                  test_set: Any,
                  test_dict: Dict[str, Test]) -> int:
    test_results = {key: [] for key in test_dict.keys()}
    show_image = True

    tests = dict.fromkeys(mechanism_fns.keys(), )
    for parent_name, mechanism_fn in mechanism_fns.items():
        tests[parent_name]['effectiveness'] = effectiveness_test(mechanism_fn, parent_name, sampling_fns[parent_name],
                                                                   classifiers)

        tests[f'{parent_name}_composition'] = composition_test(mechanism_fn, parent_name)
        if is_invertible[parent_name]:
            tests[f'{parent_name}_reversibility'] = reversibility_test(mechanism_fn, parent_name,
                                                                       sampling_fns[parent_name])

    rng = random.PRNGKey(0)
    for image, parents in test_set:
        for test_name, test_fn in test_dict.items():
            result, image_seq = test_fn(rng, image, parents)
            test_results[test_name].append(result)
            if show_image:
                plot_image_sequence(image_seq, title=test_name, n_cases=10)
        show_image = False
    test_results = {key: np.concatenate(value) for key, value in test_results.items()}
    loss_plot(test_results)
    return test_results


##


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


def loss_plot(test_results):
    for name, array in test_results.items():
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        plt.plot(mean)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=.3)
        plt.title(name)
        plt.show()
