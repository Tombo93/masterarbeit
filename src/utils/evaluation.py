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
    ) -> torch.Tensor:
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
        model.train()
        return torch.tensor(running_loss / len(test_loader.dataset))


@dataclass
class Cifar10Testing(Validation):
    loss: torch.nn.Module

    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for _, (data, labels) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                _, prediction = torch.max(logits, 1)
                metrics.update(torch.t(prediction.unsqueeze(0)), labels)

                loss = self.loss(logits, torch.squeeze(labels))
                running_loss += loss.item() * data.size(0)
        return torch.tensor(running_loss / len(test_loader.dataset))


@dataclass
class Cifar10BackdoorTesting(Validation):
    loss: torch.nn.Module
    backdoor_test_loader: DataLoader[Any]
    backdoor_metrics: Union[Metric, MetricCollection]

    def __post_init__(self):
        self.cls = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        self.cls_acc = {
            "plane": [],
            "car": [],
            "bird": [],
            "cat": [],
            "deer": [],
            "dog": [],
            "frog": [],
            "horse": [],
            "ship": [],
            "truck": [],
            "clean_data_loss": [],
        }
        self.cls_acc_backdoor = {
            "plane": [],
            "car": [],
            "bird": [],
            "cat": [],
            "deer": [],
            "dog": [],
            "frog": [],
            "horse": [],
            "ship": [],
            "truck": [],
            "backdoor_loss": [],
        }

    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> None:
        model.eval()
        with torch.no_grad():
            print("Test model on clean data...")
            class_correct, class_total = [0] * 10, [0] * 10
            test_running_loss = 0.0
            for _, (data, labels, poison_label) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                _, prediction = torch.max(logits, 1)
                metrics.update(torch.t(prediction.unsqueeze(0)), labels)

                loss = self.loss(logits, torch.squeeze(labels))
                test_running_loss += loss.item() * data.size(0)

                labels = torch.squeeze(labels)
                c = (prediction == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i in range(10):
                self.cls_acc[self.cls[i]].append(class_correct[i] / class_total[i])

            self.cls_acc["clean_data_loss"].append(test_running_loss / len(test_loader.dataset))

            print("Test model on poisoned data...")
            class_correct, class_total = [0] * 10, [0] * 10
            backdoor_running_loss = 0.0
            for _, (data, labels, poison_label) in enumerate(self.backdoor_test_loader):
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                _, prediction = torch.max(logits, 1)
                self.backdoor_metrics.update(torch.t(prediction.unsqueeze(0)), labels)

                loss = self.loss(logits, torch.squeeze(labels))
                backdoor_running_loss += loss.item() * data.size(0)

                labels = torch.squeeze(labels)
                c = (prediction == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i in range(10):
                self.cls_acc_backdoor[self.cls[i]].append(class_correct[i] / class_total[i])

            self.cls_acc_backdoor["backdoor_loss"].append(
                backdoor_running_loss / len(self.backdoor_test_loader.dataset)
            )

    def get_acc(self):
        return self.cls_acc, self.cls_acc_backdoor


class Cifar10BackdoorVal(Validation):
    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> None:
        model.eval()
        with torch.no_grad():
            print("Test model on poisoned data...")
            for _, (data, labels, poison_labels) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                poison_labels = poison_labels.to(device)

                logits = model(data)

                _, prediction = torch.max(logits, 1)
                poison_labels = torch.squeeze(poison_labels)
                # prediction = torch.t(prediction.unsqueeze(0))
                # map predicted labels to poison-labels
                prediction = torch.tensor(
                    list(map(lambda label: 1 if label == 9 else 0, prediction))
                ).to(device)
                # compare predicted vs. actual poison-labels
                metrics.update(prediction, poison_labels)
