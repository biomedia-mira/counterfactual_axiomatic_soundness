import itertools
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import Iterable
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.example_libraries.optimizers import Optimizer, OptimizerState, Params, ParamsFn
from jax.random import KeyArray
from numpy.typing import NDArray
import tensorflow_datasets as tfds
from tqdm import tqdm

from datasets.utils import image_gallery

Array = Union[jnp.ndarray, NDArray, Any]
Shape = Tuple[int, ...]
InitFn = Callable[[KeyArray, Shape], Tuple[Shape, Params]]
ApplyFn = Callable
StaxLayer = Tuple[InitFn, ApplyFn]
StaxLayerConstructor = Callable[..., StaxLayer]
UpdateFn = Callable[[int, OptimizerState, Any, KeyArray], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params, Optimizer], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitFn, ApplyFn, InitOptimizerFn]


def get_writer_fn(job_dir: Path, name: str, logging_fn: Optional[Callable[[str], None]] = None) \
        -> Callable[[Dict, int, Optional[str]], None]:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    writer = tf.summary.create_file_writer(str(logdir))

    def log_value(value: Any, tag: str, step: int) -> None:
        value = jnp.mean(value) if value.ndim <= 1 else value
        tf.summary.scalar(tag, value, step) if value.size == 1 else \
            tf.summary.image(tag, np.expand_dims(
                image_gallery(value, num_images_to_display=min(128, value.shape[0])), 0), step)
        if logging_fn is not None:
            logging_fn(f'epoch: {step:d}: \t{tag}: {value:.2f}') if value.size == 1 else None

    def log_eval(evaluation: Dict, step: int, tag: Optional[str] = None) -> None:
        with writer.as_default():
            for new_tag, value in evaluation.items():
                new_tag = tag + '/' + new_tag if tag else new_tag
                log_eval(value, step, new_tag) if isinstance(value, Dict) else log_value(value, new_tag, step)

    return log_eval


def to_numpy_iterator(data: tf.data.Dataset, batch_size: int, drop_remainder: bool = True) -> Any:
    return tfds.as_numpy(data.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE))


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.ndim == 0 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

    to_cpu = partial(jax.device_put, device=jax.devices('cpu')[0])
    new_output = jax.tree_map(to_cpu, new_output)
    return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)


def train(model: Model,
          job_dir: Path,
          seed: int,
          train_data: Iterable,
          test_data: Optional[Iterable],
          input_shape: Shape,
          optimizer: Optimizer,
          num_steps: int,
          log_every: int,
          eval_every: int,
          save_every: int) -> Params:
    model_path = job_dir / f'model.npy'
    job_dir.mkdir(exist_ok=True, parents=True)
    train_writer = get_writer_fn(job_dir, 'train')
    test_writer = get_writer_fn(job_dir, 'test')

    init_fn, apply_fn, init_optimizer_fn = model
    rng = jax.random.PRNGKey(seed)
    _, params = init_fn(rng, input_shape)
    opt_state, update, get_params = init_optimizer_fn(params, optimizer)
    for step, inputs in tqdm(enumerate(itertools.cycle(train_data)), total=num_steps):
        if step >= num_steps:
            break
        rng, _ = jax.random.split(rng)
        opt_state, loss, output = update(step, opt_state, inputs, rng)
        if step % log_every == 0:
            train_writer(output, step, None)
        if jnp.isnan(loss):
            raise ValueError('NaN loss!')

        if step % eval_every == 0 and test_data is not None:
            cum_output = None
            for test_inputs in test_data:
                rng, _ = jax.random.split(rng)
                _, output = apply_fn(get_params(opt_state), test_inputs, rng=rng)
                cum_output = accumulate_output(output, cum_output)
            test_writer(cum_output, step, None)

        if step % save_every == 0 or step == num_steps - 1:
            jnp.save(str(model_path), get_params(opt_state))

    return get_params(opt_state)
