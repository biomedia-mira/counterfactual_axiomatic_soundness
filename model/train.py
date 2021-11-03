import itertools
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental.optimizers import OptimizerState, Params, ParamsFn
from tqdm import tqdm

from components.stax_extension import Array, PRNGKey, Shape
from datasets.utils import image_gallery

InitModelFn = Callable[[PRNGKey, Shape], Params]
ApplyModelFn = Callable
UpdateFn = Callable[[int, OptimizerState, Any, PRNGKey], Tuple[OptimizerState, Array, Any]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn, ParamsFn]]
Model = Tuple[InitModelFn, ApplyModelFn, InitOptimizerFn]


def get_writer_fn(job_dir: Path, name: str, logging_fn: Optional[Callable[[str], None]] = None) \
        -> Callable[[Dict, int, Optional[str]], None]:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    writer = tf.summary.create_file_writer(str(logdir))

    def log_value(value: Any, tag: str, step: int) -> None:
        value = jnp.mean(value) if value.ndim <= 1 else value
        tf.summary.scalar(tag, value, step) if value.size == 1 else \
            tf.summary.image(tag, np.expand_dims(
                image_gallery((value * 127.5) + 127.5, num_images_to_display=min(128, value.shape[0])), 0), step)
        if logging_fn is not None:
            logging_fn(f'epoch: {step:d}: \t{tag}: {value:.2f}') if value.size == 1 else None

    def log_eval(evaluation: Dict, step: int, tag: Optional[str] = None) -> None:
        with writer.as_default():
            for new_tag, value in evaluation.items():
                new_tag = tag + '/' + new_tag if tag else new_tag
                log_eval(value, step, new_tag) if isinstance(value, Dict) else log_value(value, new_tag, step)

    return log_eval


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.ndim == 0 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

    to_cpu = partial(jax.device_put, device=jax.devices('cpu')[0])
    new_output = jax.tree_map(to_cpu, new_output)
    return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)


def train(model: Model,
          input_shape: Shape,
          job_dir: Path,
          train_data: Iterable,
          test_data: Optional[Iterable],
          num_steps: int,
          log_every: int,
          eval_every: int,
          save_every: int) -> Params:
    job_dir.mkdir(exist_ok=True, parents=True)
    init_fn, apply_fn, init_optimizer_fn = model
    train_writer = get_writer_fn(job_dir, 'train')
    test_writer = get_writer_fn(job_dir, 'test')

    rng = jax.random.PRNGKey(1234)
    _, params = init_fn(rng, input_shape)
    opt_state, update, get_params = init_optimizer_fn(params)

    for step, inputs in tqdm(enumerate(itertools.cycle(train_data)), total=num_steps):
        if step >= num_steps:
            break
        rng, _ = jax.random.split(rng)
        opt_state, loss, output = update(step, opt_state, inputs, rng)
        if step % log_every == 0:
            train_writer(output, step)
        if jnp.isnan(loss):
            raise ValueError('NaN loss')

        if step % eval_every == 0 and test_data is not None:
            cum_output = None
            for test_inputs in test_data:
                rng, _ = jax.random.split(rng)
                _, output = apply_fn(get_params(opt_state), test_inputs, rng=rng)
                cum_output = accumulate_output(output, cum_output)
            test_writer(cum_output, step)

        if step % save_every == 0:
            jnp.save(str(job_dir / f'model.npy'), get_params(opt_state))

    jnp.save(str(job_dir / f'model.npy'), get_params(opt_state))
    return get_params(opt_state)
