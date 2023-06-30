import numpy as np
import torch
from torch.utils.data import DataLoader

from dataclasses import dataclass

from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold
from utils.optimizer import OptimizationLoop
from data.dataset import Subset


@dataclass
class Experiment:
    """
    dataloader params: {
        "batch_size" : 32,
        "num_workers" : 4,
        "shuffle" : True,
        "pin_memory" : True,
    }
    """

    name: str
    params: Dict[str, Any]
    dataloader_params: Dict[str, Any]
    data: Any
    optimizer: OptimizationLoop
    kfold: StratifiedKFold
    device: torch.device

    def run(self):
        """
        if k-fold cross validation: average the calculated metrics

        """
        if self.kfold:
            self._run_kfold(self.kfold)

    def _run_kfold(self, kfold):
        for batch_size in self.params.batches:
            models = self.params.models
            for model in models:
                model.to(self.device)
                for fold, (train_indices, val_indices) in enumerate(
                    kfold.split(self.data.imgs, self.data.labels)
                ):
                    print([f"Fold {fold}"])
                    print(
                        f"train -  {np.bincount(self.data.labels[train_indices])}   \
                        |   test -  {np.bincount(self.data.labels[val_indices])}"
                    )
                    train_set = Subset(dataset=self.data, indices=train_indices)
                    val_set = Subset(dataset=self.data, indices=val_indices)
                    train_loader = DataLoader(train_set, **self.dataloader_params)
                    val_loader = DataLoader(val_set, **self.dataloader_params)
