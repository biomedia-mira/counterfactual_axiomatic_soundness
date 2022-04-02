import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from clu.metric_writers import create_default_writer
from jax.example_libraries.optimizers import Optimizer, Params
from tqdm import tqdm

from components import Model, Shape
from components.stax_extension import Array
from datasets.utils import image_gallery
from utils import flatten_nested_dict
from queue import Queue


def get_writer_fn(job_dir: Path, name: str, logging_fn: Optional[Callable[[str], None]] = None) \
        -> Callable[[Dict, int], None]:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    writer = create_default_writer(str(logdir))

    def writer_fn(evaluation: Dict, step: int) -> None:
        flat = flatten_nested_dict(evaluation)
        scalars = {'/'.join(key): jnp.mean(value) for key, value in flat.items() if value.ndim <= 1}
        images = {'/'.join(key): image_gallery(value, num_images_to_display=min(128, value.shape[0]))[np.newaxis]
                  for key, value in flat.items() if value.ndim == 4}
        writer.write_scalars(step, scalars)
        writer.write_images(step, images)
        if logging_fn is not None:
            [logging_fn(f'epoch: {step:d}: \t{key}: {value:.2f}') for key, value in scalars.items()]

    return writer_fn


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.ndim == 0 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

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
          save_every: int,
          overwrite: bool) -> Params:
    model_path = job_dir / f'model.npy'
    if model_path.exists() and not overwrite:
        return np.load(str(model_path), allow_pickle=True)
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
            train_writer(output, step)
        if jnp.isnan(loss):
            raise ValueError('NaN loss!')

        if step % eval_every == 0 and test_data is not None:
            cum_output = None
            for test_inputs in test_data:
                rng, _ = jax.random.split(rng)
                _, output = apply_fn(get_params(opt_state), test_inputs, rng=rng)
                cum_output = accumulate_output(output, cum_output)
            test_writer(cum_output, step)

        if step % save_every == 0 or step == num_steps - 1:
            jnp.save(str(model_path), get_params(opt_state))

    return get_params(opt_state)
