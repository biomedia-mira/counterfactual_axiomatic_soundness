import itertools
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.ops
from jax import jit
from torch.utils.tensorboard import SummaryWriter

from trainer.evaluation import update_eval, log_eval
from trainer.types import InitFn, ApplyFn, InitOptimizerFn, DataStream


def get_summary_writer(job_dir: Path, name: str) -> SummaryWriter:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    return SummaryWriter(str(logdir))


def train(init_fun: InitFn,
          apply_fun: ApplyFn,
          init_optimizer_fun: InitOptimizerFn,
          input_shape: Tuple[int, ...],
          job_dir: Path,
          num_epochs: int,
          train_data_stream: DataStream,
          test_data_stream: DataStream,
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
        for inputs in train_data_stream():
            opt_state, loss, outputs = update(next(itercount), opt_state, inputs)
            eval_ = update_eval(eval_, outputs)
            if jnp.isnan(loss):
                raise ValueError('NaN loss')
        log_eval(eval_, epoch, train_writer)

        if epoch % eval_every == 0:
            eval_ = None
            for inputs in test_data_stream():
                _, outputs = jit(apply_fun)(params, inputs)
                eval_ = update_eval(eval_, outputs)
            log_eval(eval_, epoch, test_writer)

        if epoch % save_every == 0:
            jnp.save(str(job_dir / 'model.np'), params)
