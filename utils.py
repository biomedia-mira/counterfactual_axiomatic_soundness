
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray


def flatten_nested_dict(nested_dict: Dict[Any, Any], key: Tuple[Any, ...] = ()) -> Dict[Any, Any]:
    new_dict: Dict[Any, Any] = {}
    for sub_key, value in nested_dict.items():
        new_key = (*key, sub_key)
        if isinstance(value, dict):
            new_dict.update(flatten_nested_dict(value, new_key))
        else:
            new_dict.update({new_key: value})
    return new_dict


def image_gallery(array: NDArray, ncols: int = 16, num_images_to_display: int = 128,
                  decode_fn: Callable[[NDArray], NDArray] = lambda x: 127.5 * x + 127.5) -> NDArray:
    array = np.clip(decode_fn(array), a_min=0, a_max=255) / 255.
    array = array[::len(array) // num_images_to_display][:num_images_to_display]
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols + int(bool(nindex % ncols))
    pad = np.zeros(shape=(nrows * ncols - nindex, height, width, intensity))
    array = np.concatenate((array, pad), axis=0)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result
