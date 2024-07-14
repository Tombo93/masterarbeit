import os
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data.dataset import NumpyDataset
from models.models import ModelFactory
from utils.optimizer import IsicTrainer
from utils.training import TrainingFactory
from utils.evaluation import TestFactory
from utils.metrics import MetricFactory, AverageMetricDict, save_metrics_to_csv
from utils.experiment import StratifierFactory


SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)


def main(cfg, save_model=False, debug=False):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    batch_size = cfg.hparams.batch_size
    epochs = cfg.hparams.epochs
    n_workers = cfg.hparams.num_workers
    # seed = cfg.hparams.rng_seed
    lr = cfg.hparams.lr
    momentum = cfg.hparams.momentum
    weight_decay = cfg.hparams.decay
    num_classes = cfg.task.num_classes
    data_path = cfg.data.data
    model_save_path = os.path.join(
        cfg.model.isic_base,
        f"Lf{cfg.task.train}-{cfg.data.id}-{cfg.hparams.id}-{datetime.datetime.now():%Y%m%d-%H%M}.pth",
    )
    report_name_train = os.path.join(
        cfg.task.reports,
        f"Lf{cfg.task.train}-{cfg.data.id}-{cfg.hparams.id}-train-{datetime.datetime.now():%Y%m%d-%H%M}",
    )
    report_name_test = os.path.join(
        cfg.task.reports,
        f"Lf{cfg.task.test}-{cfg.data.id}-{cfg.hparams.id}-test-{datetime.datetime.now():%Y%m%d-%H%M}",
    )

    training = TrainingFactory.make(cfg.task.train)
    testing = TestFactory.make(cfg.task.test)
    kfold_avg_metrics = AverageMetricDict()
    train_meter, test_meter = MetricFactory.make(cfg.task.metrics, num_classes)
    train_meter.to(device)
    test_meter.to(device)
    # model = ModelFactory().make(
    #     "resnet18",
    #     num_classes,
    #     load_from_state_dict=True,
    #     model_path=cfg.model.isic_backdoor,
    #     #     model_path=os.path.join(cfg.model.isic_base, "isic-base.pth"),
    #     random_weights=False,
    # )

    data = NumpyDataset(data_path, transforms.ToTensor(), exclude_trigger=False)
    stratifier = StratifierFactory().make(
        strat_type="multi-label", data=data, n_splits=5
    )

    for train_indices, test_indices in stratifier:
        model = ModelFactory().make("resnet18", num_classes, random_weights=True)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        trainloader = DataLoader(
            Subset(data, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        testloader = DataLoader(
            Subset(data, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        avg_train_metrics = {metric: [] for metric in train_meter.keys()}
        avg_train_metrics["Loss"] = []
        avg_val_metrics = {metric: [] for metric in test_meter.keys()}
        training = training(criterion, optimizer, trainloader)
        validation = validation(testloader)
        for _ in range(epochs):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            train_meter.update(logits, torch.squeeze(labels))
            loss = loss(logits, torch.squeeze(labels))
            _running_loss += loss.item() * data.size(0)
            avg_train_metrics["Loss"] = train_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            validation.run(model, test_meter, device)

            test = test_meter.compute()
            for metric, value in train_meter.compute().items():
                avg_train_metrics[metric].append(value.cpu().numpy())
            for metric, value in test.items():
                avg_val_metrics[metric].append(value.cpu().numpy())
            train_meter.reset()
            test_meter.reset()

        kfold_avg_metrics.add(train_dict=avg_train_metrics, val_dict=avg_val_metrics)

    if save_model:
        torch.save(model.net.state_dict(), model_save_path)
    avg_train_metrics, avg_test_metrics = kfold_avg_metrics.compute()
    save_metrics_to_csv(avg_train_metrics, report_name_train, cfg.task.train)
    save_metrics_to_csv(avg_test_metrics, report_name_test, cfg.task.test)


if __name__ == "__main__":
    main()
