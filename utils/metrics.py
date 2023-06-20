import torch

from dataclasses import dataclass, field
from torchmetrics.classification import Accuracy, AUROC, Precision

from torchmetrics.metric import Metric
from typing import List, Dict, Any


@dataclass
class AverageLoss:
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
            AverageMeter('acc', Accuracy(task='binary')),
            AverageMeter('auroc', AUROC(task='binary')),
            AverageMeter('precision', Precision(task='binary'))
        ]

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()
    
    def update(self, x: torch.TensorType, y: torch.TensorType) -> None:
        assert x.shape == y.shape
        for m in self.metrics:
            m.update(x, y)

    def compute(self) -> Dict[str, float]:
        return {m.name : m.avg for m in self.metrics}
    
    def to(self, device: torch.DeviceObjType) -> None:
        for m in self.metrics:
            m.metric.to(device)