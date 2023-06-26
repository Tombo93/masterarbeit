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
    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.DeviceObjType,
    ) -> None:
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x = x.to(device=device)
                y = y.to(device=device)
                pred = model(x)
                _, pred_labels = pred.max(dim=1)
                # metrics.update(pred_labels, y)
        model.train()


class MetricAndLossValidation(Validation):
    def __init__(self, loss_func: torch.nn.Module) -> None:
        super().__init__()
        self.loss = loss_func

    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.DeviceObjType,
    ) -> None:
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for _, (x, y) in enumerate(test_loader):
                x = x.to(device=device)
                y = y.to(device=device)
                pred = model(x)

                loss = self.loss(pred, torch.unsqueeze(y, 1).float())
                running_loss += loss.item() * x.size(0)

                # _, pred_labels = pred.max(dim=1)
                pred_labels = torch.flatten(pred)
                metrics.update(pred_labels, y)
            print(f"Validation Loss: {running_loss / len(test_loader.dataset)}")
        model.train()
        return running_loss / len(test_loader.dataset)
