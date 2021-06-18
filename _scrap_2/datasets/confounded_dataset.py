from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple, Dict

import numpy as np
import torch.autograd.functional
from torch.utils.data.dataset import Dataset


class Mechanism(ABC):
    @abstractmethod
    def __call__(self, index: int, images: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        pass


class CounfoundedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, mechanisms: Sequence[Mechanism]) -> None:
        self.base_dataset = base_dataset
        self.mechanisms = mechanisms

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        image, label = self.base_dataset.__getitem__(index)
        vars = {}
        for mechanism in self.mechanisms:
            image, vars_ = mechanism(index, image)
            vars.update(vars_)

        return image, label, vars

    def __len__(self):
        return len(self.base_dataset)
