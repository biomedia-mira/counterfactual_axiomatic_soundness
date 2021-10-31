import itertools
from typing import Callable, Dict, Iterable, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit

from components.typing import Array, Params

Test = Callable[[Array, Dict[str, Array]], Tuple[Array, Array]]
Classifier = Callable[[Array], Array]
Mechanism = Callable[[Array, Dict[str, Array], Array, Array], Tuple[Array, Array]]


def l2(x1: Array, x2: Array) -> Array:
    return np.mean(np.power(x1 - x2, 2.), axis=(1, 2, 3))


def plot_image_sequence(image_seq: np.ndarray, title: str = '', n_cases: int = 10) -> None:
    image_seq = image_seq[:n_cases]
    gallery = np.moveaxis(image_seq, 1, 2).reshape((n_cases * image_seq.shape[2],
                                                    image_seq.shape[1] * image_seq.shape[3], image_seq.shape[4]))
    plt.imshow(gallery)
    plt.title(title)
    plt.show()


def compile_fn(fn, params):
    def _fn(*args, **kwargs):
        return jit(fn)(params, *args, **kwargs)

    return _fn


def build_functions(params: Params, classifiers, divergences, mechanisms):
    _classifiers = {key: compile_fn(classifier[1], params=params[0][key]) for key, classifier in classifiers.items()}
    _divergences = {key: compile_fn(divergence[1], params=params[1][key]) for key, divergence in divergences.items()}
    _mechanisms = {key: compile_fn(mechanisms[1], params=params[2][key]) for key, mechanisms in mechanisms.items()}
    return _classifiers, _divergences, _mechanisms,


def permute_transform_test(mechanism_fns: Dict[str, Mechanism], parent_dims: Dict[str, int], noise_dim: int) -> Test:
    def test(image: Array, parents: Dict[str, Array]) -> Array:

        do_parents = {p_name: jnp.eye(p_dim)[np.random.randint(0, p_dim, size=image.shape[0])]
                      for p_name, p_dim in parent_dims.items()}
        noise = np.zeros(shape=(image.shape[0], noise_dim))

        l2_loss = []
        images_to_plot = [image, np.zeros_like(image)]
        images_to_calc = []
        for parent_order in itertools.permutations(mechanism_fns, len(mechanism_fns)):
            _image, _parents = image, parents
            for parent_name in parent_order:
                mechanism_fn = mechanism_fns[parent_name]
                _image, _ = mechanism_fn(_image, _parents, do_parents[parent_name], noise)
                _parents = {**_parents, parent_name: do_parents[parent_name]}
                images_to_plot.append(_image)
            images_to_plot.append(np.zeros_like(image))
            images_to_calc.append(_image)
            # images.append(np.zeros_like(_image))

        for im1, im2 in list(itertools.combinations(images_to_calc, 2)):
            l2_loss.append(l2(im1, im2))
            images_to_plot.append(np.abs(im1-im2))

        return np.moveaxis(np.array(l2_loss), 0, 1), np.moveaxis(np.array(images_to_plot), 0, 1)

    return test


def repeat_transform_test(mechanism_fn: Mechanism, parent_name: str, noise_dim: int, n_repeats: int = 10) -> Test:
    def test(image: Array, parents: Dict[str, Array]) -> Array:
        l2_loss = []
        noise = np.zeros(shape=(image.shape[0], noise_dim))
        transformed_image = image
        image_sequence = [transformed_image]
        for j in range(n_repeats):
            transformed_image, _ = mechanism_fn(transformed_image, parents, parents[parent_name], noise)
            l2_loss.append(l2(image, transformed_image))
            image_sequence.append(transformed_image)
        return np.moveaxis(np.array(l2_loss), 0, 1), np.moveaxis(np.array(image_sequence), 0, 1)

    return test


def cycle_transform_test(mechanism_fn: Mechanism,
                         parent_name: str,
                         noise_dim: int,
                         parent_dim: int,
                         cycle_length: int = 4,
                         num_repeats: int = 5) -> Test:
    def test(image: Array, parents: Dict[str, Array]) -> Array:
        l2_loss = []
        noise = np.zeros(shape=(image.shape[0], noise_dim))

        do_parent_cycle = jnp.eye(parent_dim)[np.random.randint(0, parent_dim, size=(cycle_length - 1, image.shape[0]))]
        do_parent_cycle = jnp.concatenate((do_parent_cycle, parents[parent_name][jnp.newaxis]))
        do_parent_cycle = jnp.concatenate([do_parent_cycle] * num_repeats, axis=0)

        transformed_image = image
        image_sequence = [transformed_image]
        transformed_parents = parents
        for do_parent in do_parent_cycle:
            transformed_image, _ = mechanism_fn(transformed_image, transformed_parents, do_parent, noise)
            l2_loss.append(l2(image, transformed_image))
            image_sequence.append(transformed_image)
            transformed_parents = {**transformed_parents, parent_name: do_parent}
        return np.moveaxis(np.array(l2_loss), 0, 1), np.moveaxis(np.array(image_sequence), 0, 1)

    return test


def loss_plot(test_results):
    for name, array in test_results.items():
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        plt.plot(mean)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=.3)
        plt.title(name)
        plt.show()


def perform_tests(test_set: Iterable[Tuple[Array, Dict[str, Array]]], test_dict: Dict[str, Test]) -> int:
    test_results = {key: [] for key in test_dict.keys()}
    show_image = True
    for inputs in test_set:
        image, parents = inputs[frozenset()]
        for test_name, test_fn in test_dict.items():
            result, image_seq = test_fn(image, parents)
            test_results[test_name].append(result)
            if show_image:
                plot_image_sequence(image_seq, title=test_name, n_cases=10)
        show_image = False
    test_results = {key: np.concatenate(value) for key, value in test_results.items()}
    loss_plot(test_results)
    return test_results
