import csv
from datetime import datetime

from abc import ABC, abstractmethod
from typing import Any, Dict


class Logger(ABC):
    @abstractmethod
    def log(self, *args, **kwargs):
        """Implement a logger"""


class SimpleLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.logfilename_train = f'{self.logdir}/logs_{datetime.now()}_train.csv'
        self.logfilename_test = f'{self.logdir}/logs_{datetime.now()}_test.csv'
        with open(self.logfilename_train, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['epoch'] +
                [key for key, _ in sorted(self.train_metrics.items(), key=lambda x: x[0])]
                )
        with open(self.logfilename_test, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['epoch'] +
                [key for key, _ in sorted(self.valid_metrics.items(), key=lambda x: x[0])]
                )
    
    def log(self,
            epoch: int,
            train_metrics: Dict[Any, Any],
            valid_metrics: Dict[Any, Any]):
        with open(self.logfilename_train, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
            [str(epoch)] +
            [value.item() for _, value in sorted(train_metrics.items(), key=lambda x: x[0])]
            )
        with open(self.logfilename_test, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(
            [epoch] +
            [value.item() for _, value in sorted(valid_metrics.items(), key=lambda x: x[0])]
            )
        