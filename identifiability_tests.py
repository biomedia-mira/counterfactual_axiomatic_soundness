import itertools
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from components import Array
from models import ClassifierFn

Test = Callable[[Array, Dict[str, Array]], Tuple[Array, Array]]
DistanceMetric = Callable[[Array, Array], Array]
# [[params, [image, parents], [image, parents]], [div_loss, output]]
DivergenceFn = Callable[[Tuple[Array, Array], Tuple[Array, Array]], Tuple[Array, Any]]
# [[params, image, parent, do_parent, do_noise], do_image]
MechanismFn = Callable[[Array, Array, Array], Array]


def l2(x1: Array, x2: Array) -> Array:
    return np.mean(np.power(x1 - x2, 2.), axis=(1, 2, 3))


def effectiveness_test(mechanism_fn: MechanismFn, parent_name: str, classifiers: Dict[str, ClassifierFn]) -> Test:
    def test(image: Array, parents: Dict[str, Array]) -> Array:
        parent = parents[parent_name]

        do_image = mechanism_fn(image, parent, do_parent)
        do_parents = {**parents, parent_name: do_parent}

        output = {}
        for _parent_name, _parent in do_parents.items():
            _, output[_parent_name] = classifiers[_parent_name]((do_image, _parent))

        return output

    return test


def composition_test(mechanism_fn: MechanismFn, parent_name: str,
                     distance_metric: DistanceMetric = l2, horizon: int = 10) -> Test:
    def test(image: Array, parents: Dict[str, Array]) -> Array:
        distance_loss = []
        parent = parents[parent_name]
        do_image = image
        image_sequence = [do_image]
        for j in range(horizon):
            do_image = mechanism_fn(do_image, parent, parent)
            distance_loss.append(distance_metric(image, do_image))
            image_sequence.append(do_image)
        return np.moveaxis(np.array(distance_loss), 0, 1), np.moveaxis(np.array(image_sequence), 0, 1)

    return test


def reversibility_test(mechanism_fn: MechanismFn, parent_dim: int, parent_name: str, cycle_length: int = 2,
                       num_cycles: int = 1) -> Test:
    def test(image: Array, parents: Dict[str, Array]) -> Array:
        l2_loss = []
        do_parent_cycle = jnp.eye(parent_dim)[np.random.randint(0, parent_dim, size=(cycle_length - 1, image.shape[0]))]
        do_parent_cycle = jnp.concatenate((do_parent_cycle, parents[parent_name][jnp.newaxis]))
        do_parent_cycle = jnp.concatenate([do_parent_cycle] * num_cycles, axis=0)

        do_image = image
        image_sequence = [do_image]
        do_parents = parents
        for do_parent in do_parent_cycle:
            do_image = mechanism_fn(do_image, do_parents[parent_name], do_parent)
            l2_loss.append(l2(image, do_image))
            image_sequence.append(do_image)
            do_parents = {**do_parents, parent_name: do_parent}
        return np.moveaxis(np.array(l2_loss), 0, 1), np.moveaxis(np.array(image_sequence), 0, 1)

    return test


def perform_tests(mechanism_fns: Dict[str, MechanismFn],
                  test_set: Any,
                  test_dict: Dict[str, Test]) -> int:
    test_results = {key: [] for key in test_dict.keys()}
    show_image = True

    tests = {}
    for parent_name, mechanism_fn in mechanism_fns.items():
        tests[f'{parent_name}_effectiveness'] = composition_test(mechanism_fn, parent_name)
        tests[f'{parent_name}_composition'] = composition_test(mechanism_fn, parent_name)

        tests[f'{parent_name}_reversibility'] = reversibility_test(mechanism_fn, parent_dim, parent_name)


    for image, parents in test_set:
        for mechanism in mechamisms.values():

        for test_name, test_fn in test_dict.items():

            result, image_seq = test_fn(image, parents)
            test_results[test_name].append(result)
            if show_image:
                plot_image_sequence(image_seq, title=test_name, n_cases=10)
        show_image = False
    test_results = {key: np.concatenate(value) for key, value in test_results.items()}
    loss_plot(test_results)
    return test_results


##

def plot_image_sequence(image_seq: np.ndarray, title: str = '', n_cases: int = 10) -> None:
    image_seq = image_seq[:n_cases]
    gallery = np.moveaxis(image_seq, 1, 2).reshape((n_cases * image_seq.shape[2],
                                                    image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4]))
    plt.imshow(gallery)
    plt.title(title)
    plt.show()


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