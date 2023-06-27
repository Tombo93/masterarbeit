import torch

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Union
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric


class Validation(ABC):
    @abstractmethod
    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> Union[float, None]:
        """Implement a validation loop"""


class MetricValidation(Validation):
    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> None:
        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(test_loader):
                x = x.to(device=device)
                y = y.to(device=device)
                pred = model(x)
                _, pred_labels = pred.max(dim=1)
                metrics.update(pred_labels, y)
        model.train()


@dataclass
class MetricAndLossValidation(Validation):
    loss: torch.nn.Module

    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> float:
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for _, (x, y) in enumerate(test_loader):
                x = x.to(device=device)
                y = y.to(device=device)
                pred = model(x)

                loss = self.loss(pred, y.float())
                running_loss += loss.item() * x.size(0)

                metrics.update(pred, y)
            print(f"Validation Loss: {running_loss / len(test_loader.dataset)}")
        model.train()
        return running_loss / len(test_loader.dataset)
