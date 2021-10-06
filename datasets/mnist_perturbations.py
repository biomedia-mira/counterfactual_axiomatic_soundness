from typing import Callable, Dict, Tuple

import numpy as np
import tensorflow as tf
from skimage import draw, morphology, transform

from datasets.morphomnist import skeleton
from datasets.morphomnist.morpho import ImageMorphology

tf.config.experimental.set_visible_devices([], 'GPU')


def get_swell_fn(strength: float = 3, radius: float = 7):
    def warp(xy: np.ndarray, morph: ImageMorphology, strength, radius) -> np.ndarray:
        loc_sampler = skeleton.LocationSampler()
        centre = loc_sampler.sample(morph)[::-1]
        radius = (radius * np.sqrt(morph.mean_thickness) / 2.) * morph.scale
        offset_xy = xy - centre
        distance = np.hypot(*offset_xy.T)
        weight = (distance / radius) ** (strength - 1)
        weight[distance > radius] = 1.
        return centre + weight[:, None] * offset_xy

    @tf_numpy_fn
    def swell(image: np.ndarray):
        assert image.ndim == 3 and image.shape[-1] == 1
        morph = ImageMorphology(image[..., 0])
        return np.expand_dims(transform.warp(image[..., 0], lambda xy: warp(xy, morph, strength, radius)), axis=-1)

    return swell


def get_fracture_fn(thickness: float = 1.5, prune: float = 2, num_frac: int = 3):
    _ANGLE_WINDOW = 2
    _FRAC_EXTENSION = .5

    def _endpoints(morph, centre):
        angle = skeleton.get_angle(morph.skeleton, *centre, _ANGLE_WINDOW * morph.scale)
        length = morph.distance_map[centre[0], centre[1]] + _FRAC_EXTENSION * morph.scale
        angle += np.pi / 2.  # Perpendicular to the skeleton
        normal = length * np.array([np.sin(angle), np.cos(angle)])
        p0 = (centre + normal).astype(int)
        p1 = (centre - normal).astype(int)
        return p0, p1

    def _draw_line(img, p0, p1, brush):
        h, w = brush.shape
        ii, jj = draw.line(*p0, *p1)
        for i, j in zip(ii, jj):
            img[i:i + h, j:j + w] &= brush

    @tf_numpy_fn
    def fracture(image: np.ndarray) -> np.ndarray:
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
        return np.expand_dims(frac_img[r:-r, r:-r], axis=-1)

    return fracture

