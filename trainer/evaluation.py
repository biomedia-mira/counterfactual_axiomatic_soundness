from typing import Dict, Callable, Any, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def tree_apply(tree: Dict, fun: Callable[[Any], Any]) -> Dict:
    for key, value in tree.items():
        if isinstance(value, dict):
            tree[key] = tree_apply(value, fun)
        else:
            tree[key] = fun(value)
    return tree


def tree_op(tree_1: Dict, tree_2: Dict, op_fun: Callable[[Any, Any], Any]) -> Dict:
    for key, value in tree_1.items():
        if isinstance(value, dict):
            tree_1[key] = tree_op(value, tree_2[key], op_fun)
        else:
            tree_1[key] = op_fun(value, tree_2[key])
    return tree_1


def tree_flatten(tree: Dict, parent_key: str = '', sep: str = '/') -> Dict:
    items: List = []
    for key, value in tree.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, Dict):
            items.extend(tree_flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def log_value(value: Any, tag: str, step: int, writer: SummaryWriter,
              logging_fn: Callable[[str], None] = print) -> None:
    message = f'epoch: {step:d}:'
    if value.size == 1:
        writer.add_scalar(tag, value.item(), step)
        message += f'\t{tag}: {value.item():.2f}'
    elif value.size > 1 and value.ndim == 1:
        writer.add_scalar(tag, np.mean(value), step)
        message += f'\t{tag}: {np.mean(value):.2f}'
    elif value.ndim == 4 and value.shape[1] == 3:
        writer.add_image(tag, torchvision.utils.make_grid(torch.tensor(value)), step)
    logging_fn(message)


def to_py(x: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else x.to_py()


def update_value(value: np.ndarray, new_value: jnp.ndarray) -> np.ndarray:
    if value.size == 1:
        return value + to_py(new_value)
    elif value.size > 1 and value.ndim == 1:
        return np.concatenate((value, to_py(new_value)))
    elif value.ndim == 4 and value.shape[1] == 3:
        return to_py(new_value)
    raise ValueError('Unsupported input!')


def update_eval(eval_: Optional[Dict], outputs: Dict) -> Dict:
    return tree_apply(outputs, lambda x: to_py(x)) if eval_ is None else tree_op(eval_, outputs, update_value)


def log_eval(evaluation: Optional[Dict], epoch: int, writer: SummaryWriter) -> None:
    assert isinstance(evaluation, dict)
    for key, value in tree_flatten(evaluation).items():
        log_value(value, key, epoch, writer)
