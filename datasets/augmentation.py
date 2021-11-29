from typing import Tuple

import tensorflow as tf


def tf_randint(minval: int, maxval: int, shape: Tuple = ()) -> tf.Tensor:
    return tf.random.uniform(minval=minval, maxval=maxval, dtype=tf.int32, shape=shape)


def random_crop_and_rescale(image: tf.Tensor, fractions: Tuple[float, float] = (.2, .2)) -> tf.Tensor:
    shape = image.shape[:-1]
    start = tuple(tf_randint(minval=0, maxval=int(s * fpd / 2.)) for s, fpd in zip(shape, fractions))
    stop = tuple(tf_randint(minval=int(s * (1. - fpd / 2.)), maxval=s) for s, fpd in zip(shape, fractions))
    slices = tuple((slice(_start, _stop) for _start, _stop in zip(start, stop)))
    cropped_image = image[slices]
    return tf.image.resize(cropped_image, size=shape)
