from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple, Dict

import torch.autograd.functional
from torch.utils.data.dataset import Dataset


class Mechanism(ABC):
    @abstractmethod
    def __call__(self, index: int, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class CounfoundedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, mechanisms: Sequence[Mechanism]) -> None:
        self.base_dataset = base_dataset
        self.mechanisms = mechanisms

    def __getitem__(self, index: int) -> Tuple[Any, Any, List[Any]]:
        image, label = self.base_dataset.__getitem__(index)
        vars = {}
        for mechanism in self.mechanisms:
            image, vars_ = mechanism(index, image)
            vars.update(vars_)

        return image, label, vars

    def __len__(self):
        return len(self.base_dataset)
