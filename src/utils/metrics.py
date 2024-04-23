from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import numpy as np
import pandas as pd
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    Precision,
    Recall,
    BinaryRecall,
    MulticlassConfusionMatrix,
)
from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)


class CustomBinaryRecall(BinaryRecall):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="none",
            multidim_average=self.multidim_average,
        )


class MetricFactory:
    @staticmethod
    def make(task, num_classes):
        match task:
            case "diagnosis":
                return (
                    MetricCollection(
                        [Accuracy(task="multiclass", num_classes=num_classes)]
                    ),
                    MetricCollection(
                        [
                            Accuracy(task="multiclass", num_classes=num_classes),
                            AUROC(task="multiclass", num_classes=num_classes),
                            MulticlassConfusionMatrix(num_classes, normalize="true"),
                        ]
                    ),
                )
            case "family_history":
                return (
                    MetricCollection([Accuracy(task="binary")]),
                    MetricCollection(
                        [
                            Accuracy(task="binary"),
                            AUROC(task="binary"),
                            Precision(task="binary"),
                            Recall(task="binary"),
                        ]
                    ),
                )
            case "backdoor":
                return (
                    MetricCollection(
                        [Accuracy(task="multiclass", num_classes=num_classes)]
                    ),
                    MetricCollection(
                        [
                            Accuracy(task="binary"),
                            CustomBinaryRecall(),  # Recall(task="binary"),
                            Precision(task="binary"),
                            AUROC(task="binary"),
                        ]
                    ),
                )
            case _:
                raise NotImplementedError(
                    f"The metrics you're trying to use for this task ({task}) haven't been implemented yet.\n\
                        Available tasks: [ diagnosis , family_history , backdoor ]"
                )


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


def save_metrics_to_csv(metrics, report_path):
    df = pd.DataFrame(metrics)
    df.to_csv(report_path)
