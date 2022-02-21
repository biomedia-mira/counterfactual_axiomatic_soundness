from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from components.stax_extension import Array
from datasets.utils import image_gallery


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


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.ndim == 0 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

    to_cpu = partial(jax.device_put, device=jax.devices('cpu')[0])
    new_output = jax.tree_map(to_cpu, new_output)
    return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)
