import itertools
import os
import pickle
from typing import Sequence, Tuple, Dict, Callable, Generator, Any
import json
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from pathlib import Path

import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)


def get_colorize_fun(cm: np.ndarray, labels: np.ndarray):
    color_indices = np.array(list(map(lambda label: np.random.choice(cm.shape[1], p=cm[label]), labels)))
    colors = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (.5, 0, 0),
                       (0, .5, 0), (0, 0, .5)))

    def colorize(index: int, data) -> Tuple[np.ndarray, Dict[str, int]]:
        image, parents = data
        color_idx = int(color_indices[index])
        return np.stack([image] * 3, axis=0) * colors[color_idx].reshape(3, 1, 1), {'color': color_idx}

def f(dataset, mechanisms):
    dataset = dataset.map(lambda image, label: (image, {'digit': label}))
    dataset = dataset.enumerate()
    for mechanism in mechanisms:
        dataset = dataset.map(mechanism, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(ds_info.splits['train'].num_examples)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset





class ConfoundedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, mechanisms: Sequence[Callable]) -> None:
        self.base_dataset = base_dataset
        self.mechanisms = mechanisms
        self.cache_dir = Path('/vol/biomedic/users/mm6818/projects/grand_canyon/data/confounded_mnist')

    def save(self, image_path: Path, parents_path: Path, image: np.ndarray, parents: Dict[str, np.ndarray]) -> None:
        image_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(image_path, image)
        with open(parents_path, 'w') as fp:
            json.dump(parents, fp)

    def load(self, image_path: Path, parents_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        image = np.load(image_path)
        with open(parents_path, 'r') as fp:
            parents = json.load(fp)
        return image, parents

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        image_path = self.cache_dir / f'{index:d}' / 'image.npy'
        parents_path = self.cache_dir / f'{index:d}' / 'parents.json'
        if image_path.exists() and parents_path.exists():
            return self.load(image_path, parents_path)

        image, label = self.base_dataset.__getitem__(index)
        image = np.array(image) / 255.
        parents = {'label': label}
        for mechanism in self.mechanisms:
            image, parents_ = mechanism(index, image)
            parents.update(parents_)
        print('3')
        self.save(image_path, parents_path, image, parents)

        return image, parents

    def __len__(self):
        return len(self.base_dataset)


def get_uncounfounded_dataloaders(dataset: ConfoundedDataset,
                                  parent_dims: Dict[str, int],
                                  dataloader_kwargs: Dict):
    parents = {key: np.array(list(map(lambda x: x[1][key], iter(dataset)))) for key, dim in parent_dims.items()}
    indicator = {key: [parents[key] == i for i in range(dim)] for key, dim in parent_dims.items()}
    index_map = np.array([np.logical_and.reduce(a) for a in itertools.product(*indicator.values())])
    index_map = index_map.reshape((*parent_dims.values(), -1))
    counts = np.sum(index_map, axis=-1)
    joint_distribution = counts / np.sum(counts)

    dataloaders = {}
    marginals = {}
    for axis, parent in enumerate(parents.keys()):
        marginal_distribution = np.sum(joint_distribution, axis=tuple(set(range(counts.ndim)) - {axis}), keepdims=True)
        distribution = marginal_distribution * np.sum(joint_distribution, axis=axis, keepdims=True)
        weights = distribution / counts
        sample_weights = np.sum(weights[..., np.newaxis] * index_map, axis=tuple(range(len(parents))))
        sampler = WeightedRandomSampler(sample_weights, len(dataset), generator=torch.Generator())
        marginals[parent] = np.squeeze(marginal_distribution)
        dataloaders[parent] = DataLoader(dataset, **dataloader_kwargs, sampler=sampler)

    return dataloaders, marginals


def create_data_stream_fun(dataset: ConfoundedDataset,
                           parent_dims: Dict[str, int],
                           dataloader_kwargs: Dict) -> Tuple[
    Callable[..., Generator[Any, None, None]], Dict[str, np.ndarray]]:
    def one_hot(dim: int, value: torch.Tensor) -> np.ndarray:
        return np.eye(dim)[value.numpy()]

    def collate_fn(batch):
        image, parents = default_collate(batch)
        image_np = image.numpy()
        parents_np = {parent: one_hot(parent_dims[parent], value) for parent, value in parents.items()}
        return image_np, parents_np

    dataloader_kwargs.update({'collate_fn': collate_fn})
    joint_dataloader = DataLoader(dataset, shuffle=True, **dataloader_kwargs)
    unconfounded_dataloaders, marginals = get_uncounfounded_dataloaders(dataset, parent_dims, dataloader_kwargs)

    def data_stream() -> Generator[Any, None, None]:
        for inputs in tqdm(zip(joint_dataloader, *unconfounded_dataloaders.values())):
            yield {'joint': inputs[0], **{key: inputs[i] for i, key in enumerate(unconfounded_dataloaders.keys())}}

    return data_stream, marginals
