import torch
from torch.cuda.amp import autocast, GradScaler

from abc import ABC, abstractmethod
from typing import Any, Union
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader


class Training(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """Implement a training loop"""


class BasicTraining(Training):
    """
    train_loader: DataLoader[Any],
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Union[Metric, MetricCollection],
    device: torch.DeviceObjType) -> None:
    """
    def run(self,
            train_loader: DataLoader[Any],
            model: torch.nn.Module,
            loss_func: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            metrics: Union[Metric, MetricCollection],
            device: torch.DeviceObjType) -> None:
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            prediction = model(data)
            loss = loss_func(prediction, torch.unsqueeze(labels, 1).float())

            _, pred_labels = prediction.max(dim=1)
            metrics.update(pred_labels, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class ScaledMixedPrecisionTraining(Training):
    def run(self,
            train_loader: DataLoader[Any],
            model: torch.nn.Module,
            loss_func: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            metrics: Union[Metric, MetricCollection],
            scaler: GradScaler,
            device: torch.DeviceObjType) -> None:
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            with autocast():
                prediction = model(data)
                loss = loss_func(prediction, labels)

            _, pred_labels = prediction.max(dim=1)
            metrics.update(pred_labels, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
