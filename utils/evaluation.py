import torch

from abc import ABC, abstractmethod
from typing import Any, Union
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric


class Validation(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """Implement a validation loop"""


class MetricValidation(Validation):
    def run(self,
            test_loader: DataLoader[Any],
            model: torch.nn.Module,
            metrics: Union[Metric, MetricCollection],
            device: torch.DeviceObjType) -> None:
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x = x.to(device=device)
                y = y.to(device=device)
                pred = model(x)
                _, pred_labels = pred.max(dim=1)
                metrics.update(pred_labels, y)   
        model.train()
