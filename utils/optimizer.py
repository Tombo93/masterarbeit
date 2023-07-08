import torch
from torch.utils.tensorboard.writer import SummaryWriter

from dataclasses import dataclass
from typing import Union, Any
from utils.training import Training
from utils.evaluation import Validation
from torch.utils.data import DataLoader
from utils.logger import Logger


@dataclass
class OptimizationLoop:
    """Executes the optimization loop:

    Procedure
    --------
    1. training
    2. validation
    3. recording metrics
    """

    model: torch.nn.Module
    training: Training
    validation: Validation
    train_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    train_metrics: Any
    val_metrics: Any
    epochs: int
    device: torch.device
    logdir: str
    logger: Union[Logger, None] = None

    def __post_init__(self):
        if self.logger is None:
            self.writer = SummaryWriter(
                log_dir=self.logdir,
            )

    def optimize(self) -> None:
        for epoch in range(self.epochs):
            train_loss = self.training.run(
                self.train_loader, self.model, self.train_metrics, self.device
            )
            valid_loss = self.validation.run(
                self.test_loader, self.model, self.val_metrics, self.device
            )
            total_train_metrics = self.train_metrics.compute()
            total_valid_metrics = self.val_metrics.compute()
            print(f"Training metrics for epoch {epoch}: {total_train_metrics}")
            print(f"Validation metrics for epoch {epoch}: {total_valid_metrics}")

            if self.logger is None:
                for metric, value in total_train_metrics.items():
                    self.writer.add_scalar(f"Train/{metric}", value, epoch)
                for metric, value in total_valid_metrics.items():
                    self.writer.add_scalar(f"Test/{metric}", value, epoch)

                self.writer.add_scalar(f"Train/Loss", train_loss, epoch)
                self.writer.add_scalar(f"Test/Loss", valid_loss, epoch)

            if self.logger is not None:
                self.logger.log(epoch, total_train_metrics, total_valid_metrics)

            self.train_metrics.reset()
            self.val_metrics.reset()

    def overfit_batch_test(
        self,
        loss_func: torch.nn.Module,
        optim: torch.optim.Optimizer,
        n_batches: int,
        batch_size: int,
    ) -> None:
        """Train on n_batches

        Parameters
        ----------
        loss_func : torch.nn.Module
        optim : torch.optim.Optimizer
        n_batches : int
        batch_size : int
        """
        train_data = [next(iter(self.train_loader)) for _ in range(n_batches)]
        valid_data = [next(iter(self.test_loader)) for _ in range(n_batches)]

        for epoch in range(self.epochs):
            # Training Phase
            running_loss = 0.0
            for data, labels in train_data:
                data = data.to(self.device)
                labels = labels.to(self.device)
                prediction = self.model(data)
                loss = loss_func(prediction, torch.unsqueeze(labels, 1).float())

                running_loss += loss.item() * data.size(0)

                # _, pred_labels = prediction.max(dim=1)
                self.train_metrics.update(torch.flatten(prediction), labels)

                loss.backward()
                optim.step()
                optim.zero_grad()
            train_loss = running_loss / (len(train_data) * batch_size)
            print(f"Training Loss: {train_loss}")

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                running_loss = 0.0
                for data, labels in train_data:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    prediction = self.model(data)
                    loss = loss_func(prediction, torch.unsqueeze(labels, 1).float())
                    running_loss += loss.item() * data.size(0)

                    # _, pred_labels = pred.max(dim=1)
                    self.val_metrics.update(torch.flatten(prediction), labels)
                eval_loss = running_loss / (len(valid_data) * batch_size)
                print(f"Validation Loss: {eval_loss}")
            self.model.train()

            # Metrics
            total_train_metrics = self.train_metrics.compute()
            total_valid_metrics = self.val_metrics.compute()
            print(f"Training metrics for epoch {epoch}: {total_train_metrics}")
            print(f"Validation metrics for epoch {epoch}: {total_valid_metrics}")

            if self.logger is None:
                for metric, value in total_train_metrics.items():
                    self.writer.add_scalar(f"Train/{metric}", value, epoch)
                for metric, value in total_valid_metrics.items():
                    self.writer.add_scalar(f"Test/{metric}", value, epoch)
                self.writer.add_scalar(f"Train/Loss", train_loss, epoch)
                self.writer.add_scalar(f"Test/Loss", eval_loss, epoch)

            if self.logger is not None:
                self.logger.log(epoch, total_train_metrics, total_valid_metrics)

            self.train_metrics.reset()
            self.val_metrics.reset()
