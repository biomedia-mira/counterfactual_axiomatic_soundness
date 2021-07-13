import itertools
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.experimental.optimizers import OptimizerState, Params

T = TypeVar('T')
Tree = Union[Dict[str, T], Dict[str, 'Tree']]
Shape = Tuple[int, ...]
InitFn = Callable[[jnp.ndarray, Shape], Tuple[Shape, Params]]
ApplyFn = Callable[[Params, Tree[jnp.ndarray]], Tuple[jnp.ndarray, Tree[jnp.ndarray]]]
UpdateFn = Callable[[int, OptimizerState, Tree[jnp.ndarray]], Tuple[OptimizerState, jnp.ndarray, Tree[jnp.ndarray]]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn]]


def get_summary_writer(job_dir: Path, name: str) -> tf.summary.SummaryWriter:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    return tf.summary.create_file_writer(str(logdir))


def train(init_fun: InitFn,
          apply_fun: ApplyFn,
          init_optimizer_fun: InitOptimizerFn,
          update_eval,
          log_eval,
          input_shape: Tuple[int, ...],
          job_dir: Path,
          num_epochs: int,
          train_data: Iterable,
          test_data: Optional[Iterable],
          eval_every: int,
          save_every: int) -> None:
    train_writer = get_summary_writer(job_dir, 'train')
    test_writer = get_summary_writer(job_dir, 'test')

    rng = jax.random.PRNGKey(0)
    params = init_fun(rng, input_shape)
    opt_state, update = init_optimizer_fun(params)

    itercount = itertools.count()
    for epoch in range(num_epochs):
        eval_ = None
        for i, inputs in enumerate(train_data):
            opt_state, loss, outputs = update(next(itercount), opt_state, inputs, rng)
            rng, _ = jax.random.split(rng)
            eval_ = update_eval(eval_, outputs)
            if jnp.isnan(loss):
                raise ValueError('NaN loss')
        log_eval(eval_, epoch, train_writer)

        if epoch % eval_every == 0 and test_data is not None:
            eval_ = None
            for inputs in test_data:
                _, outputs = jax.jit(apply_fun)(params, inputs)
                eval_ = update_eval(eval_, outputs)
            log_eval(eval_, epoch, test_writer)

        if epoch % save_every == 0:
            jnp.save(str(job_dir / 'model.np'), params)
