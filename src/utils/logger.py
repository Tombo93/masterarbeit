import csv
import numpy as np
from datetime import datetime

from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass


class Logger(ABC):
    @abstractmethod
    def log(self, *args, **kwargs):
        """Implement a logger"""


class SimpleLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.logfilename_train = f"{self.logdir}/logs_{datetime.now()}_train.csv"
        self.logfilename_test = f"{self.logdir}/logs_{datetime.now()}_test.csv"
        with open(self.logfilename_train, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch"]
                + [
                    key
                    for key, _ in sorted(self.train_metrics.items(), key=lambda x: x[0])
                ]
            )
        with open(self.logfilename_test, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch"]
                + [
                    key
                    for key, _ in sorted(self.valid_metrics.items(), key=lambda x: x[0])
                ]
            )

    def log(
        self, epoch: int, train_metrics: Dict[Any, Any], valid_metrics: Dict[Any, Any]
    ):
        with open(self.logfilename_train, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [str(epoch)]
                + [
                    value.item()
                    for _, value in sorted(train_metrics.items(), key=lambda x: x[0])
                ]
            )
        with open(self.logfilename_test, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch]
                + [
                    value.item()
                    for _, value in sorted(valid_metrics.items(), key=lambda x: x[0])
                ]
            )


@dataclass
class LoggerWrapper:
    logger: Any
    epochs: int
    lr: float
    batch_size: int
    device: str
    filename: str

    def log_start_experiment(self) -> None:
        self.logger.info("Experiment")
        self.logger.info("Metadata")
        self.logger.info("----------")
        self.logger.info(
            f"Epochs: {self.epochs} | lrs: {self.lr} | batch_sizes: {self.batch_size} | device: {self.device} | data used: {self.filename}"
        )
        self.logger.info("----------")

    def log_single_fold(self, fold, labels, train_indices, val_indices) -> None:
        self.logger.info(f"Fold {fold}")
        self.logger.info(
            f"train -  {np.bincount(labels[train_indices])}   |   test -  {np.bincount(labels[val_indices])}"
        )
        self.logger.info(f"lr: {self.lr} | batch_size: {self.batch_size}")
        self.logger.info("Indices of fold")
        self.logger.info("Train")
        self.logger.info("-----------------------------")
        self.logger.info(f"{' '.join(map(str, train_indices))}")
        self.logger.info("-----------------------------")
        self.logger.info("Validation")
        self.logger.info("-----------------------------")
        self.logger.info(f"{' '.join(map(str, val_indices))}")
