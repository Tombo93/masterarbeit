from abc import ABC, abstractmethod
from typing import Any, Union
from dataclasses import dataclass

import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader


class Training(ABC):
    @abstractmethod
    def run(
        self,
        train_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> Union[float, None]:
        """Implement a training loop"""


class BasicTraining(Training):
    def __init__(
        self, loss_func: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.loss = loss_func
        self.optim = optimizer

    def run(
        self,
        train_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> None:
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


@dataclass
class ScaledMixedPrecisionTraining(Training):
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: GradScaler

    def run(
        self,
        train_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> None:
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


@dataclass
class PlotLossTraining(Training):
    """Class for handling the training loop.

    Parameters
    ----------
    loss : torch.nn.Module
        A torch loss function
    optimizer : torch.optim.Optimizer

    Methods
    ----------
    run() : runs the training
    """

    loss: torch.nn.Module
    optim: torch.optim.Optimizer

    def run(
        self,
        train_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """Runs the training loop

        Parameters
        ----------
        train_loader : DataLoader[Any]
            For the training split
        model : torch.nn.Module
            The Neural Network
        metrics : Any
            A torchmetrics Object or MetricCollection
        device : torch.DeviceObjType

        Returns
        -------
        float
            The loss of one epoch
        """
        running_loss = 0.0
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            prediction = model(data)
            loss = self.loss(prediction, labels.float())
            running_loss += loss.item() * data.size(0)
            metrics.update(prediction, labels)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return torch.tensor(running_loss / len(train_loader.dataset))


@dataclass
class Cifar10Training(Training):

    loss: torch.nn.Module
    optim: torch.optim.Optimizer

    def run(
        self,
        train_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Any,
        device: torch.device,
    ) -> torch.Tensor:
        print("Training model...")
        running_loss = 0.0
        model.train()
        for _, (data, labels, _) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            _, prediction = torch.max(logits, 1)
            metrics.update(torch.t(prediction.unsqueeze(0)), labels)

            loss = self.loss(logits, torch.squeeze(labels))
            running_loss += loss.item() * data.size(0)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return torch.tensor(running_loss / len(train_loader.dataset))


@dataclass
class IsicTraining(Training):

    loss: torch.nn.Module
    optim: torch.optim.Optimizer

    def run(self, train_loader, model, metrics, device):
        print("Training model...")
        running_loss = 0.0
        model.train()
        for data, labels, _, _ in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            _, prediction = torch.max(logits, 1)
            metrics.update(torch.t(prediction.unsqueeze(0)), labels)

            loss = self.loss(logits, torch.squeeze(labels))
            running_loss += loss.item() * data.size(0)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return torch.tensor(running_loss / len(train_loader.dataset))
