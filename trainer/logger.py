from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from clu.metric_writers import create_default_writer

from components.stax_extension import Array
from datasets.utils import image_gallery
from utils import flatten_nested_dict


def get_writer_fn(job_dir: Path, name: str, logging_fn: Optional[Callable[[str], None]] = None) \
        -> Callable[[Dict, int], None]:
    logdir = (job_dir / 'logs' / name)
    logdir.mkdir(exist_ok=True, parents=True)
    writer = create_default_writer(str(logdir))

    def log_eval(evaluation: Dict, step: int) -> None:
        flat = flatten_nested_dict(evaluation)
        scalars = {key: jnp.mean(value) for key, value in flat.items() if value.ndim <= 1}
        images = {key: image_gallery(value, num_images_to_display=min(128, value.shape[0]))
                  for key, value in flat.items() if value.ndim == 4}
        writer.write_scalars(step, scalars)
        writer.write_images(step, images)
        writer.flush()

        # if logging_fn is not None:
        #     logging_fn(f'epoch: {step:d}: \t{tag}: {value:.2f}') if value.size == 1 else None

    return log_eval


def accumulate_output(new_output: Any, cum_output: Optional[Any]) -> Any:
    def update_value(value: Array, new_value: Array) -> Array:
        return value + new_value if value.ndim == 0 \
            else (jnp.concatenate((value, new_value)) if value.ndim == 1 else new_value)

    return new_output if cum_output is None else jax.tree_multimap(update_value, cum_output, new_output)
