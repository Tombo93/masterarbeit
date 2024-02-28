import torch
import numpy as np

from dataclasses import dataclass
from torchmetrics.classification import Accuracy, AUROC, Precision

from torchmetrics.metric import Metric
from typing import List, Dict, Any


class AverageMetricDict:
    def __init__(self) -> None:
        self.train_meter_dicts = []
        self.val_meter_dicts = []

    def add(self, train_dict, val_dict):
        self.train_meter_dicts.append(train_dict)
        self.val_meter_dicts.append(val_dict)

    def compute_single(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
        return mean_dict

    def compute(self):
        return self.compute_single(self.train_meter_dicts), self.compute_single(
            self.val_meter_dicts
        )


@dataclass
class AverageMetric:
    name: str
    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.name, self.avg


@dataclass
class AverageMeter:
    name: str
    metric: Metric
    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, preds: torch.TensorType, target: torch.TensorType) -> None:
        self.val = self.metric(preds, target)
        self.sum += self.val * target.numel()
        self.count += target.numel()
        self.avg = self.sum / self.count


class AverageMeterCollection:
    def __init__(self, phase: str) -> None:
        self.phase: str = phase
        self.metrics: List[Any] = [
            AverageMeter("acc", Accuracy(task="binary")),
            AverageMeter("auroc", AUROC(task="binary")),
            AverageMeter("precision", Precision(task="binary")),
        ]

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    def update(self, x: torch.TensorType, y: torch.TensorType) -> None:
        assert x.shape == y.shape
        for m in self.metrics:
            m.update(x, y)

    def compute(self) -> Dict[str, float]:
        return {m.name: m.avg for m in self.metrics}

    def to(self, device: torch.DeviceObjType) -> None:
        for m in self.metrics:
            m.metric.to(device)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    avg_meter = AverageMetricDict()
    avg_meter.add(
        train_dict={"BinaryAUROC": [1, 1, 1, 1], "Precision": [1, 1, 1, 1]},
        val_dict={"BinaryAUROC": [1, 1, 2, 1], "Precision": [1, 1, 1, 1]},
    )
    avg_meter.add(
        train_dict={"BinaryAUROC": [2, 2, 2, 2], "Precision": [1, 1, 1, 1]},
        val_dict={"BinaryAUROC": [1, 1, 1, 1], "Precision": [1, 1, 1, 4]},
    )
    avg_meter.add(
        train_dict={"BinaryAUROC": [1, 2, 3, 4], "Precision": [1, 1, 1, 1]},
        val_dict={"BinaryAUROC": [1, 1, 2, 1], "Precision": [1, 1, 3, 1]},
    )
    avg_train_metrics, avg_val_metrics = avg_meter.compute()
    print(avg_train_metrics, avg_val_metrics)
    for k, v in avg_val_metrics.items():
        plt.figure(k)
        plt.title(f"{k}")
        plt.xlabel("epoch")
        # plt.ylabel("Some unit")
        xs = range(0, len(v))
        ys = v
        plt.plot(xs, ys, "-.")
        plt.savefig(f"Test-{k}.png")
