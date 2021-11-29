from typing import Callable, Dict, Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from skimage import draw, morphology, transform

from datasets.confounding import IMAGE, Mechanism
from datasets.morphomnist import skeleton
from datasets.morphomnist.morpho import ImageMorphology


def function_dict_to_mechanism(function_dict: Dict[int, Callable[[IMAGE], IMAGE]], cm: NDArray[np.float_]) -> Mechanism:
    def apply_fn(image: IMAGE, confounder: int) -> Tuple[IMAGE, int]:
        idx = np.random.choice(cm.shape[1], p=cm[confounder])
        return function_dict[idx](image), idx

    return apply_fn


def get_thickening_fn(amount: float = 1.2) -> Callable[[IMAGE], IMAGE]:
    def apply_fn(image: IMAGE) -> IMAGE:
        morph = ImageMorphology(image[..., 0])
        radius = int(amount * morph.scale * morph.mean_thickness / 2.)
        return np.array(np.expand_dims(morphology.dilation(image[..., 0], morphology.disk(radius)), axis=-1))

    return apply_fn


def get_thinning_fn(amount: float = .7) -> Callable[[IMAGE], IMAGE]:
    def apply_fn(image: IMAGE) -> IMAGE:
        morph = ImageMorphology(image[..., 0])
        radius = int(amount * morph.scale * morph.mean_thickness / 2.)
        return np.array(np.expand_dims(morphology.erosion(image[..., 0], morphology.disk(radius)), axis=-1))

    return apply_fn


def get_swell_fn(strength: float = 3, radius: float = 7) -> Callable[[IMAGE], IMAGE]:
    def _warp(xy: IMAGE, morph: ImageMorphology) -> IMAGE:
        loc_sampler = skeleton.LocationSampler()
        centre = loc_sampler.sample(morph)[::-1]
        _radius = (radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale
        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / _radius) ** (strength - 1)
        weight[distance > _radius] = 1.
        return np.array(centre + weight[:, None] * offset_xy)

    def swell(image: IMAGE) -> IMAGE:
        assert image.ndim == 3 and image.shape[-1] == 1
        morph = ImageMorphology(image[..., 0])
        return np.array(np.expand_dims(transform.warp(image[..., 0], lambda xy: _warp(xy, morph)), axis=-1))

    return swell


def get_fracture_fn(thickness: float = 1.5, prune: float = 2, num_frac: int = 3) -> Callable[[IMAGE], IMAGE]:
    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def _endpoints(morph: ImageMorphology, centre: NDArray) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
        angle = skeleton.get_angle(morph.skeleton, *centre, _ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + _FRAC_EXTENSION * morph.scale
        angle += np.pi / 2.  # Perpendicular to the skeleton
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    def _draw_line(img: IMAGE, p0: NDArray[np.int_], p1: NDArray[np.int_], brush: NDArray[np.bool_]) -> None:
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush

    def fracture(image: IMAGE) -> IMAGE:
        morph = ImageMorphology(image[..., 0])
        loc_sampler = skeleton.LocationSampler(prune, prune)

        up_thickness = thickness * morph.scale
        r = int(np.ceil((up_thickness - 1) / 2))
        brush = ~morphology.disk(r).astype(bool)
        frac_img = np.pad(image[..., 0], pad_width=r, mode='constant', constant_values=False)
        try:
            centres = loc_sampler.sample(morph, num_frac)
        except ValueError:  # Skeleton vanished with pruning, attempt without
            centres = skeleton.LocationSampler().sample(morph, num_frac)
        for centre in centres:
            p0, p1 = _endpoints(morph, centre)
            _draw_line(frac_img, p0, p1, brush)
        return np.array(np.expand_dims(frac_img[r:-r, r:-r], axis=-1))

    return fracture


def get_colorize_fn(cm: NDArray[np.float_]) -> Mechanism:
    colors = tf.constant(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                          (0, 1, 1), (1, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5)))

    def apply_fn(image: IMAGE, confounder: int) -> Tuple[IMAGE, int]:
        idx = np.random.choice(cm.shape[1], p=cm[confounder])
        color = colors[idx]
        return np.array(np.repeat(image, 3, axis=-1) * np.array(color)).astype(np.uint8), int(idx)

    return apply_fn
