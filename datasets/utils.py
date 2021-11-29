from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from datasets.confounding import apply_mechanisms_to_dataset, Mechanism


def load_cached_dataset(dataset_dir: Path, dataset: tf.data.Dataset, mechanisms: List[Mechanism],
                        parent_dims: Dict[str, int]) -> Tuple[tf.data.Dataset, Dict[str, NDArray]]:
    parents_path = str(dataset_dir / 'parents.npy')
    images_path = str(dataset_dir / 'images.npy')
    try:
        images = np.load(images_path)
        parents = np.load(parents_path, allow_pickle=True).item()
    except FileNotFoundError:
        print('Dataset not found, creating new copy...')
        images, parents = apply_mechanisms_to_dataset(dataset, mechanisms, parent_dims)
        dataset_dir.mkdir(exist_ok=True, parents=True)
        np.save(images_path, images)
        np.save(parents_path, parents)
    dataset = tf.data.Dataset.from_tensor_slices((images, parents))
    return dataset, parents


def image_gallery(array: np.ndarray, ncols: int = 16, num_images_to_display: int = 128) -> np.ndarray:
    array = np.clip(array, a_min=0, a_max=255) / 255.
    array = array[::len(array) // num_images_to_display]
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols + int(bool(nindex % ncols))
    pad = np.zeros(shape=(nrows * ncols - nindex, height, width, intensity))
    array = np.concatenate((array, pad), axis=0)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result
