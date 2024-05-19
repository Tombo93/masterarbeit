import os
import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import pandas as pd

from data.dataset import IsicBackdoorDataset, NumpyDataset
from utils.optimizer import IsicTrainer, BackdoorTrainer
from utils.training import IsicTraining
from utils.evaluation import TestFactory, IsicBackdoor
from utils.metrics import MetricFactory, AverageMetricDict, save_metrics_to_csv
from models.models import ModelFactory
from utils.experiment import StratifierFactory


SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)


def main(cfg, debug=False):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = cfg.hparams.batch_size
    epochs = cfg.hparams.epochs
    n_workers = cfg.hparams.num_workers
    seed = cfg.hparams.rng_seed
    lr = cfg.hparams.lr
    momentum = cfg.hparams.momentum
    weight_decay = cfg.hparams.decay
    num_classes = len(cfg.data.diagnosis)
    poison_class = cfg.data.poison_encoding
    clean_data_path = cfg.data.data

    report_name_train = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-train-{datetime.datetime.now():%Y%m%d-%H%M}.csv",
    )
    report_name_test = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-test-{datetime.datetime.now():%Y%m%d-%H%M}.csv",
    )

    backdoor_data = IsicBackdoorDataset(
        cfg.backdoor.data, transforms.ToTensor(), poison_class
    )
    clean_data = NumpyDataset(clean_data_path, transforms.ToTensor())

    model = ModelFactory().make(
        "resnet18",
        num_classes,
        load_from_state_dict=True,
        model_path=os.path.join(cfg.model.isic_base, "isic-base.pth"),
    )
    model.to(device)

    backdoor_test, diagnosis_test = TestFactory.make("backdoor")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    train_meter, test_meter, diag_test_meter = MetricFactory.make(
        "backdoor", num_classes
    )
    train_meter.to(device)
    test_meter.to(device)
    diag_test_meter.to(device)
    kfold_avg_metrics = AverageMetricDict(n_meters=["train", "test", "diag_test"])
    stratifier = StratifierFactory().make(
        strat_type="multi-label", data=backdoor_data, n_splits=5
    )
    for train_indices, test_indices in stratifier:
        backdoor_trainloader = DataLoader(
            Subset(backdoor_data, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        backdoor_testloader = DataLoader(
            Subset(backdoor_data, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        clean_data_testloader = DataLoader(
            Subset(clean_data, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        train_test_handler = IsicTrainer(
            model=model,
            training=IsicTraining(criterion, optimizer, backdoor_trainloader),
            validation=backdoor_test(backdoor_testloader, poison_class),
            trainmetrics=train_meter,
            testmetrics=test_meter,
            epochs=epochs,
            device=device,
        )
        train_test_handler.optimize(debug=debug)
        train_metrics, test_metrics, diag_test_metrics = (
            train_test_handler.get_metrics()
        )
        kfold_avg_metrics.add_meters(
            {
                "train": train_metrics,
                "test": test_metrics,
                "diag_test": diag_test_metrics,
            }
        )

    avg_train_metrics, avg_test_metrics = kfold_avg_metrics.compute()
    df = pd.DataFrame(avg_train_metrics)
    df.to_csv(report_name_train)
    df = pd.DataFrame(avg_test_metrics)
    df.to_csv(report_name_test)


if __name__ == "__main__":
    main()
