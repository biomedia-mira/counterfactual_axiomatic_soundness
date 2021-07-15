import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental.optimizers import OptimizerState, Params
from datasets.confounded_mnist import image_gallery

from components.typing import Array, Shape

InitModelFn = Callable[[Array, Shape], Params]
ApplyModelFn = Callable
UpdateFn = Callable[[int, OptimizerState, Any, jnp.ndarray], Tuple[OptimizerState, jnp.ndarray, Any]]
InitOptimizerFn = Callable[[Params], Tuple[OptimizerState, UpdateFn]]
AccumulateOutputFn = Callable[[Any, Optional[Any]], Any]
LogOutputFn = Callable[[Any], None]
Model = Tuple[InitModelFn, ApplyModelFn, InitOptimizerFn, AccumulateOutputFn, LogOutputFn]


def log_eval(evaluation: Dict, step: int, writer: tf.summary.SummaryWriter, tag: Optional[str] = None) -> None:
    for new_tag, value in evaluation.items():
        new_tag = tag + '/' + new_tag if tag else new_tag
        log_eval(value, step, writer, new_tag) if isinstance(value, Dict) else log_value(value, new_tag, step, writer)


def log_value(value: Any, tag: str, step: int, writer: tf.summary.SummaryWriter,
              logging_fn: Callable[[str], None] = print) -> None:
    with writer.as_default():
        tf.summary.scalar(tag, value, step) if value.size == 1 else tf.summary.image(tag, np.expand_dims(image_gallery(value), 0), step)
    logging_fn(f'epoch: {step:d}: \t{tag}: {value:.2f}') if value.size == 1 else None


def get_summary_writer(job_dir: Path, name: str) -> tf.summary.SummaryWriter:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    return tf.summary.create_file_writer(str(logdir))


def train(model: Model,
          input_shape: Shape,
          job_dir: Path,
          num_epochs: int,
          train_data: Iterable,
          test_data: Optional[Iterable],
          eval_every: int,
          save_every: int) -> None:
    init_fun, apply_fun, init_optimizer_fun, accumulate_output, log_output = model
    train_writer = get_summary_writer(job_dir, 'train')
    test_writer = get_summary_writer(job_dir, 'test')

    rng = jax.random.PRNGKey(0)
    params = init_fun(rng, input_shape)
    opt_state, update = init_optimizer_fun(params)

    itercount = itertools.count()
    # for epoch in range(num_epochs):
    #     cum_output = None
    #     for i, inputs in enumerate(train_data):
    #         opt_state, loss, output = update(next(itercount), opt_state, inputs, rng)
    #         rng, _ = jax.random.split(rng)
    #         cum_output = accumulate_output(output, cum_output)
    #         if jnp.isnan(loss):
    #             raise ValueError('NaN loss')
    #     log_eval(log_output(cum_output), epoch, train_writer)
    #
    #     if epoch % eval_every == 0 and test_data is not None:
    #         cum_output = None
    #         for inputs in test_data:
    #             _, output = jax.jit(apply_fun)(params, inputs)
    #             cum_output = accumulate_output(output, cum_output)
    #         log_eval(log_output(cum_output), epoch, test_writer)
    #
    #     if epoch % save_every == 0:
    #         jnp.save(str(job_dir / 'model.np'), params)

    for epoch in range(num_epochs):
        for i, inputs in enumerate(train_data):
            j=next(itercount)
            opt_state, loss, output = update(j, opt_state, inputs, rng)
            rng, _ = jax.random.split(rng)
            log_eval(log_output(output), j, train_writer)
            if jnp.isnan(loss):
                raise ValueError('NaN loss')


