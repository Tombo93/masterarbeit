from abc import ABC, abstractmethod
from typing import Any, Union
from dataclasses import dataclass

import torch
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader


class Training_(ABC):
    @abstractmethod
    def run(
        self,
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> Union[float, None]:
        """Implement a training loop"""


@dataclass
class BaseTraining(Training_):

    loss: torch.nn.Module
    optim: torch.optim.Optimizer
    dl: DataLoader[Any]
    _running_loss: float = 0.0

    def train(self, data, labels, model, metrics, device):
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        _, prediction = torch.max(logits, 1)
        metrics.update(torch.t(prediction.unsqueeze(0)), labels)
        loss = self.loss(logits, torch.squeeze(labels))
        self._running_loss += loss.item() * data.size(0)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def run(self, model, metrics, device):
        self.reset_running_loss()

        model.train()
        for data, labels, _, _ in self.dl:
            self.train(data, labels, model, metrics, device)

        return self.get_running_loss()

    def reset_running_loss(self):
        self._running_loss = 0.0

    def get_running_loss(self):
        return torch.tensor(self._running_loss / len(self.dl.dataset))


@dataclass
class Cifar10Training(BaseTraining): ...


@dataclass
class IsicTraining(BaseTraining): ...


@dataclass
class IsicFXTraining(BaseTraining):
    def run(self, model, metrics, device):
        self.reset_running_loss()
        model.train()
        for data, _, extra_labels, _ in self.dl:
            self.train(data, extra_labels, model, metrics, device)
        return self.get_running_loss()


class TrainingFactory:
    @staticmethod
    def make(task):
        match task:
            case "diagnosis":
                return IsicTraining
            case "family_history":
                return IsicFXTraining
            case _:
                raise NotImplementedError(
                    f"The training you're trying to run for this task ({task}) hasn't been implemented yet.\n\
                        Available tasks: [ diagnosis , family_history ]"
                )
