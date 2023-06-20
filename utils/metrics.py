from dataclasses import dataclass
from torchmetrics.classification import Accuracy, AUROC, Precision

from torchmetrics.metric import Metric
from typing import List, Dict, Any


@dataclass
class AverageMeter:
    name: str
    val: float = 0
    avg: float = 0
    sum: float = 0
    count: int = 0
    meter: Any = None

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
class AverageMeterCollection:
    accuracy: Metric = Accuracy(task='binary')
    auroc: Metric = AUROC(task='binary')
    precision: Metric = Precision(task='binary')
    metrics: Dict[str, Metric]

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()
    
    def update(self) -> None:
        for m in self.metrics:
            m.update()