from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from clu.metric_writers import create_default_writer
from jax.tree_util import tree_map
from tqdm import tqdm
from utils import flatten_nested_dict, image_gallery

from staxplus.types import Array, ArrayTree, GradientTransformation, Model, Params, Shape


def get_writer_fn(job_dir: Path, name: str, logging_fn: Optional[Callable[[str], None]] = None) \
        -> Callable[[Dict[Any, Any], int], None]:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    writer = create_default_writer(str(logdir))

    def writer_fn(evaluation: Dict[Any, Any], step: int) -> None:
        flat = flatten_nested_dict(evaluation)
        scalars = {'/'.join(key): jnp.mean(value)
                   for key, value in flat.items() if value.ndim <= 1}
        images = {'/'.join(key): image_gallery(value, num_images_to_display=min(128, value.shape[0]))[np.newaxis]
                  for key, value in flat.items() if value.ndim == 4}
        writer.write_scalars(step, scalars)
        writer.write_images(step, images)
        if logging_fn is not None:
            [logging_fn(f'epoch: {step:d}: \t{key}: {value:.2f}')
             for key, value in scalars.items()]

    return writer_fn


def update_value(value: Array, new_value: Array) -> Array:
    return value + new_value if value.ndim == 0 \
        else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    return new_output if cum_output is None else tree_map(update_value, cum_output, new_output)


def train(model: Model,
          job_dir: Path,
          seed: int,
          train_data: Iterable[ArrayTree],
          test_data: Optional[Iterable[ArrayTree]],
          input_shape: Shape,
          optimizer: GradientTransformation,
          num_steps: int,
          log_every: int,
          eval_every: int,
          save_every: int,
          overwrite: bool,
          use_jit: bool = True) -> Params:
    model_path = job_dir / 'model.npy'
    if model_path.exists() and not overwrite:
        return np.load(str(model_path), allow_pickle=True)
    job_dir.mkdir(exist_ok=True, parents=True)
    train_writer = get_writer_fn(job_dir, 'train')
    test_writer = get_writer_fn(job_dir, 'test')
    init_fn, apply_fn, update_fn = model
    if use_jit:
        update_fn = jax.jit(update_fn, static_argnames='optimizer')
        apply_fn = jax.jit(apply_fn)
    rng = jax.random.PRNGKey(seed)
    params = init_fn(rng, input_shape)
    opt_state = optimizer.init(params)
    for step, inputs in tqdm(enumerate(train_data), total=num_steps):
        if step >= num_steps:
            break
        rng, _ = jax.random.split(rng)
        params, opt_state, loss, output = update_fn(params, optimizer, opt_state, rng, inputs)
        if step % log_every == 0:
            train_writer(output, step)
        if jnp.isnan(loss):
            raise ValueError('NaN loss!')

        if step > 0 and step % eval_every == 0 and test_data is not None:
            cum_output = None
            for test_inputs in test_data:
                rng, _ = jax.random.split(rng)
                _, output = apply_fn(params, rng, test_inputs)
                cum_output = accumulate_output(output, cum_output)
            test_writer(cum_output, step)

        if step % save_every == 0 or step == num_steps - 1:
            jnp.save(str(model_path), params)

    return params
