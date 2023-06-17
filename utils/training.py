from typing import Protocol
import torch
from torch.cuda.amp import autocast


class Training(Protocol):
    def run(self, dataloader, model, loss_func, optimizer, metrics, device):
        pass


class BasicTraining:
    """
    teststring
    """
    def run(self, train_loader, model, loss_func, optimizer, metrics, device):
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            prediction = model(data)
            loss = loss_func(prediction, torch.unsqueeze(labels, 1).float())

            _, pred_labels = prediction.max(dim=1)
            metrics.update(pred_labels, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class ScaledMixedPrecisionTraining:
    def run(self, train_loader, model, loss_func, optimizer, metrics, scaler, device):
        for _, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            with autocast():
                prediction = model(data)
                loss = loss_func(prediction, labels)

            _, pred_labels = prediction.max(dim=1)
            metrics.update(pred_labels, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


def basic_training_loop(train_loader, model, loss_func, optimizer, metrics, device):
    for _, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        prediction = model(data)
        loss = loss_func(prediction, torch.unsqueeze(labels, 1).float())
        
        _, pred_labels = prediction.max(dim=1)
        metrics.update(pred_labels, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
