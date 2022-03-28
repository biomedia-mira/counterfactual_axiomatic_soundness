import itertools
from pathlib import Path
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries.optimizers import Optimizer, Params
from tqdm import tqdm

from components import Model, Shape
from trainer.logger import accumulate_output, get_writer_fn


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
          overwrite: bool = False) -> Params:
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
