from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.metric import Metric


class Validation_(ABC):
    @abstractmethod
    def run(
        self,
        test_loader: DataLoader[Any],
        model: torch.nn.Module,
        metrics: Union[Metric, MetricCollection],
        device: torch.device,
    ) -> Union[float, None]:
        """Implement a validation loop"""


@dataclass
class Cifar10Testing(Validation_):
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
class Cifar10BackdoorTesting(Validation_):
    loss: torch.nn.Module
    backdoor_test_loader: DataLoader[Any]
    backdoor_metrics: Union[Metric, MetricCollection]

    def __post_init__(self):
        self.cls = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
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

            self.cls_acc["clean_data_loss"].append(
                test_running_loss / len(test_loader.dataset)
            )

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
                self.cls_acc_backdoor[self.cls[i]].append(
                    class_correct[i] / class_total[i]
                )

            self.cls_acc_backdoor["backdoor_loss"].append(
                backdoor_running_loss / len(self.backdoor_test_loader.dataset)
            )

    def get_acc(self):
        return self.cls_acc, self.cls_acc_backdoor


class Cifar10BackdoorVal(Validation_):
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
                prediction = torch.tensor(
                    list(map(lambda label: 1 if label == 1 else 0, prediction))
                ).to(device)
                metrics.update(prediction, poison_labels)


class _IsicBackdoor(Validation_):
    def __init__(self, dl, poison_class):
        self.dl = dl
        self.poison_class = poison_class
        self.poison_label = 1
        self.clean_label = 0
        self.class_recall = {
            0: {"tp": 0, "fn": 0},
            1: {"tp": 0, "fn": 0},
            2: {"tp": 0, "fn": 0},
            3: {"tp": 0, "fn": 0},
            4: {"tp": 0, "fn": 0},
            5: {"tp": 0, "fn": 0},
            6: {"tp": 0, "fn": 0},
        }

    def get_class_predictions(self, fx_preds, fx_labels, diag_labels):
        true_positives = torch.logical_and(fx_preds == fx_labels, fx_labels == 1)
        false_negatives = torch.logical_and(fx_preds != fx_labels, fx_labels == 1)
        for cls in self.class_recall.keys():
            mask = torch.eq(diag_labels, cls).squeeze()
            idx = torch.nonzero(mask).squeeze()
            self.class_recall[cls]["tp"] += torch.sum(true_positives[idx]).cpu().numpy()
            self.class_recall[cls]["fn"] += (
                torch.sum(false_negatives[idx]).cpu().numpy()
            )

    def compute(self):
        epsilon = 1e-7
        return {
            k: v["tp"] / (v["tp"] + v["fn"] + epsilon)
            for k, v in self.class_recall.items()
        }

    def _evaluate(self, model, metrics, device, data, diag_labels, labels):
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        _, prediction = torch.max(logits, 1)
        labels = torch.squeeze(labels)
        prediction = torch.tensor(
            list(
                map(
                    lambda label: (
                        self.poison_label
                        if label == self.poison_class
                        else self.clean_label
                    ),
                    prediction,
                )
            )
        ).to(device)
        self.get_class_predictions(prediction, labels, diag_labels)

        # metrics.update(prediction, labels)

    def evaluate(self, model, metrics, device, data, labels):
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        _, prediction = torch.max(logits, 1)
        labels = torch.squeeze(labels)

        prediction = torch.tensor(
            list(
                map(
                    lambda label: (
                        self.poison_label
                        if label == self.poison_class
                        else self.clean_label
                    ),
                    prediction,
                )
            )
        ).to(device)

        metrics.update(prediction, labels)

    def run(self, model, metrics, device):
        model.eval()
        with torch.no_grad():
            for data, diag_labels, fx_labels, _ in self.dl:
                self._evaluate(model, metrics, device, data, diag_labels, fx_labels)

    def run_debug(self, model, metrics, device, test_run_size=3):
        model.eval()
        i = 0
        with torch.no_grad():
            for data, diag_labels, fx_labels, _ in self.dl:
                self._evaluate(model, metrics, device, data, diag_labels, fx_labels)
                i += 1
                if i == test_run_size:
                    break


@dataclass
class IsicBackdoor(Validation_):

    dl: DataLoader[Any]
    poison_class: int
    poison_label: int = 1
    clean_label: int = 0

    def evaluate(self, model, metrics, device, data, labels):
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data)
        _, prediction = torch.max(logits, 1)
        labels = torch.squeeze(labels)

        prediction = torch.tensor(
            list(
                map(
                    lambda label: (
                        self.poison_label
                        if label == self.poison_class
                        else self.clean_label
                    ),
                    prediction,
                )
            )
        ).to(device)

        metrics.update(prediction, labels)

    def run(self, model, metrics, device):
        model.eval()
        with torch.no_grad():
            for data, _, fx_labels, _ in self.dl:
                self.evaluate(model, metrics, device, data, fx_labels)

    def run_debug(self, model, metrics, device, test_run_size=3):
        model.eval()
        i = 0
        with torch.no_grad():
            for data, _, fx_labels, _ in self.dl:
                self.evaluate(model, metrics, device, data, fx_labels)
                i += 1
                if i == test_run_size:
                    break


@dataclass
class IsicDiagnosis(Validation_):

    dl: DataLoader[Any]

    def run(self, model, metrics, device):
        model.eval()
        with torch.no_grad():
            for data, labels, _, _ in self.dl:
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                metrics.update(logits, torch.squeeze(labels))

    def run_debug(self, model, metrics, device, test_run_size=3):
        model.eval()
        with torch.no_grad():
            i = 0
            for data, labels, _, _ in self.dl:
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                metrics.update(logits, torch.squeeze(labels))
                i += 1
                if i == test_run_size:
                    break


@dataclass
class IsicFamilyHistory(Validation_):

    dl: DataLoader[Any]

    def run(self, model, metrics, device):
        model.eval()
        with torch.no_grad():
            for data, _, fx_labels, _ in self.dl:
                data = data.to(device)
                fx_labels = fx_labels.to(device)
                logits = model(data)
                _, prediction = torch.max(logits, 1)
                metrics.update(prediction, torch.squeeze(fx_labels))


class TestFactory:
    @staticmethod
    def make(task):
        match task:
            case "diagnosis":
                return IsicDiagnosis
            case "family_history":
                return IsicFamilyHistory
            case "backdoor":
                return IsicBackdoor
            case _:
                raise NotImplementedError(
                    f"The Testing you're trying to run for this task ({task}) \
                        hasn't been implemented yet.\n\
                        Available tasks: [ diagnosis , backdoor ]"
                )
