import os
import datetime
import random

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
from utils.experiment import StratifierFactory, write_folds_to_file


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
    lr = cfg.hparams.lr
    momentum = cfg.hparams.momentum
    weight_decay = cfg.hparams.decay
    num_classes = cfg.task.num_classes
    data_path = cfg.data.data
    model_save_path = os.path.join(
        cfg.model.isic_base,
        f"{cfg.task.train}-{cfg.data.id}-{cfg.hparams.id}-{datetime.datetime.now():%Y%m%d-%H%M}.pth",
    )
    report_name_train = os.path.join(
        cfg.task.reports,
        f"{cfg.task.train}-{cfg.data.id}-{cfg.hparams.id}-train-{datetime.datetime.now():%Y%m%d-%H%M}-10percent",
    )
    report_name_test = os.path.join(
        cfg.task.reports,
        f"{cfg.task.test}-{cfg.data.id}-{cfg.hparams.id}-test-{datetime.datetime.now():%Y%m%d-%H%M}-10percent",
    )

    training = TrainingFactory.make(cfg.task.train)
    testing = TestFactory.make(cfg.task.test)
    kfold_avg_metrics = AverageMetricDict()
    train_meter, test_meter = MetricFactory.make(cfg.task.metrics, num_classes)
    train_meter.to(device)
    test_meter.to(device)

    exclude_trigger = True if cfg.task.train == "diagnosis" else False
    data = NumpyDataset(
        data_path, transforms.ToTensor(), exclude_trigger=exclude_trigger
    )
    # write_folds_to_file(data, exclude_trigger=True)
    # return
    stratifier = StratifierFactory().make(
        strat_type="from-file",
        data=data,
        n_splits=5,
        from_file="poison10percent",
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
        train_test_handler = IsicTrainer(
            model=model,
            training=training(criterion, optimizer, trainloader),
            validation=testing(testloader),
            trainmetrics=train_meter,
            testmetrics=test_meter,
            epochs=epochs,
            device=device,
        )
        train_test_handler.optimize(debug=debug)
        train_metrics, test_metrics = train_test_handler.get_metrics()
        kfold_avg_metrics.add(train_dict=train_metrics, val_dict=test_metrics)

    if save_model:
        torch.save(model.net.state_dict(), model_save_path)
    avg_train_metrics, avg_test_metrics = kfold_avg_metrics.compute()
    save_metrics_to_csv(avg_train_metrics, report_name_train, cfg.task.train)
    save_metrics_to_csv(avg_test_metrics, report_name_test, cfg.task.test)


if __name__ == "__main__":
    main()
