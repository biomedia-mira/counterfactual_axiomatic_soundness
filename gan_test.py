import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from trainer.evaluation import get_evaluation_update_and_log_fns
from model.gan import gan_model
from trainer.training import train


def rgb_decode_fn(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255., a_min=0, a_max=255).astype(dtype=np.int32)


def rgb_encode_fn(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(image, dtype=tf.float32) / 255.


def get_dataset(batch_size):
    dataset, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)
    dataset = dataset.map(lambda image, target: (rgb_encode_fn(image), tf.one_hot(target, 10)))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=60000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return tfds.as_numpy(dataset), (-1, 28, 28, 1)


if __name__ == '__main__':
    overwrite = True
    num_epochs = 10000
    job_dir = Path('/tmp/test_run_4')
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)

    dataset, input_shape = get_dataset(2048)
    img_encode_fn, img_decode_fn = rgb_encode_fn, rgb_decode_fn

    init_fun, apply_fun, init_optimizer_fun = gan_model()
    update_eval, log_eval = get_evaluation_update_and_log_fns(img_decode_fn)

    if not job_dir.exists() or overwrite:
        train(init_fun=init_fun,
              apply_fun=apply_fun,
              init_optimizer_fun=init_optimizer_fun,
              update_eval=update_eval,
              log_eval=log_eval,
              input_shape=input_shape,
              job_dir=job_dir,
              num_epochs=num_epochs,
              train_data=dataset,
              test_data=None,
              eval_every=10,
              save_every=10)
