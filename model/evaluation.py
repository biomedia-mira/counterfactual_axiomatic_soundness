from typing import Any, Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.ops
import numpy as np
import tensorflow as tf


def tree_flatten(tree: Dict, parent_key: str = '', sep: str = '/') -> Dict:
    items: List = []
    for key, value in tree.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, Dict):
            items.extend(tree_flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def image_gallery(array: np.ndarray, ncols: int = 8):
    array = np.clip(array, a_min=0, a_max=255) / 255.
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols + int(bool(nindex % ncols))
    pad = np.zeros(shape=(nrows * ncols - nindex, height, width, intensity))
    array = np.concatenate((array, pad), axis=0)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def to_py(x: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else x.to_py()


def update_value(value: np.ndarray, new_value: jnp.ndarray) -> np.ndarray:
    if value.size == 1:
        return value + to_py(new_value)
    elif value.size > 1 and value.ndim == 1:
        return np.concatenate((value, to_py(new_value)))
    elif value.ndim == 3:
        return to_py(new_value)
    raise ValueError('Unsupported input!')


def update_eval(eval_: Optional[Dict], outputs: Dict) -> Dict:
    return jax.tree_map(to_py, outputs) if eval_ is None else jax.tree_multimap(update_value, eval_, outputs)


def get_evaluation_update_and_log_fns(decode_fn: Callable[[np.array], np.array]):
    def log_eval(evaluation: Optional[Dict], epoch: int, writer: tf.summary.SummaryWriter) -> None:
        assert isinstance(evaluation, dict)
        for key, value in tree_flatten(evaluation).items():
            log_value(value, key, epoch, writer)

    def log_value(value: Any, tag: str, step: int, writer: tf.summary.SummaryWriter,
                  logging_fn: Callable[[str], None] = print) -> None:
        message = f'epoch: {step:d}:'
        with writer.as_default():
            if value.size == 1:
                tf.summary.scalar(tag, value.item(), step)
                message += f'\t{tag}: {value.item():.2f}'
            elif value.size > 1 and value.ndim == 1:
                tf.summary.scalar(tag, np.mean(value), step)
                message += f'\t{tag}: {np.mean(value):.2f}'
            elif value.ndim == 3:
                tf.summary.image(tag, np.expand_dims(image_gallery(decode_fn(value)), axis=0), step)
            logging_fn(message)

    return update_eval, log_eval
