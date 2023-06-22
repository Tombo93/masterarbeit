import torch
from torch.cuda.amp import autocast, GradScaler

from abc import ABC, abstractmethod
from typing import Any, Union
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader

from utils.metrics import AverageMeterCollection, AverageMeter

class Training(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """Implement a training loop"""


class BasicTraining(Training):
    def __init__(self,
                 loss_func: torch.nn.Module,
                 optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self.loss = loss_func
        self.optim = optimizer

    def run(self,
            train_loader: DataLoader[Any],
            model: torch.nn.Module,
            metrics: Union[Metric, MetricCollection],
            device: torch.DeviceObjType) -> None:
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            prediction = model(data)
            loss = self.loss(prediction, torch.unsqueeze(labels, 1).float())

            _, pred_labels = prediction.max(dim=1)
            metrics.update(pred_labels, labels)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()


class ScaledMixedPrecisionTraining(Training):
    def __init__(self,
                 loss_func: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scaler: GradScaler) -> None:
        super().__init__()
        self.loss = loss_func
        self.optim = optimizer
        self.scaler = scaler

    def run(self,
            train_loader: DataLoader[Any],
            model: torch.nn.Module,
            metrics: Union[Metric, MetricCollection],
            device: torch.DeviceObjType) -> None:
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            with autocast():
                prediction = model(data)
                loss = self.loss(prediction, labels)

            _, pred_labels = prediction.max(dim=1)
            metrics.update(pred_labels, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()


class PlotLossTraining(Training):
    def __init__(self,
                 loss_func: torch.nn.Module,
                 optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self.loss = loss_func
        self.optim = optimizer

    def run(self,
            train_loader: DataLoader[Any],
            model: torch.nn.Module,
            metrics: Any,
            device: torch.DeviceObjType) -> float:
        
        running_loss = 0.0
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            prediction = model(data)
            loss = self.loss(prediction, torch.unsqueeze(labels, 1).float())            
            running_loss += loss.item() * data.size(0)

            pred_labels = torch.flatten(prediction)
            metrics.update(pred_labels, labels)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return running_loss / len(train_loader.dataset)
