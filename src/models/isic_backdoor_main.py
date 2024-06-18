import os
import datetime
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import pandas as pd

from data.dataset import IsicBackdoorDataset, NumpyDataset
from utils.optimizer import IsicTrainer, BackdoorTrainer, IsicBackdoorTrainer
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
        device = torch.device("cuda:1")
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

    # report_name_train = os.path.join(
    #     cfg.backdoor.reports,
    #     f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-train-{datetime.datetime.now():%Y%m%d-%H%M}.csv",
    # )
    report_name = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-{datetime.datetime.now():%Y%m%d-%H%M}",
    )

    backdoor_data = IsicBackdoorDataset(
        cfg.backdoor.data, transforms.ToTensor(), poison_class
    )
    clean_data = NumpyDataset(clean_data_path, transforms.ToTensor())

    backdoor_test, diagnosis_test = TestFactory.make("backdoor")

    train_meter, test_meter, diag_test_meter = MetricFactory.make(
        "backdoor", num_classes
    )
    train_meter.to(device)
    test_meter.to(device)
    diag_test_meter.to(device)
    kfold_avg_metrics = AverageMetricDict(n_meters=["train", "test", "diag_test"])

    backdoor_stratifier = StratifierFactory().make(
        strat_type="multi-label", data=backdoor_data, n_splits=5
    )

    export_model = None
    for train_indices, test_indices in backdoor_stratifier:
        model = ModelFactory().make(
            "resnet18", num_classes, load_from_state_dict=False, random_weights=True
        )
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
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
        train_test_handler = IsicBackdoorTrainer(
            model,
            {
                "train": {
                    "c": IsicTraining(criterion, optimizer, backdoor_trainloader),
                    "metrics": train_meter,
                },
                "test": {
                    "c": backdoor_test(backdoor_testloader, poison_class),
                    "metrics": test_meter,
                },
            },
            ["train", "test"],
            epochs,
            device,
        )
        train_test_handler.optimize(debug=debug)
        kfold_avg_metrics.add_meters(train_test_handler.get_metrics())
        export_model = copy.deepcopy(model)

    clean_stratifier = StratifierFactory().make(
        strat_type="multi-label", data=clean_data, n_splits=5
    )
    for train_indices, test_indices in clean_stratifier:
        model = copy.deepcopy(export_model)
        clean_data_testloader = DataLoader(
            Subset(clean_data, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        diag_test_handler = IsicBackdoorTrainer(
            model,
            {
                "diag_test": {
                    "c": diagnosis_test(clean_data_testloader),
                    "metrics": diag_test_meter,
                },
            },
            ["diag_test"],
            epochs,
            device,
        )
        diag_test_handler.optimize(debug=debug)
        kfold_avg_metrics.add_meters(diag_test_handler.get_metrics())

    for component_name, meters in kfold_avg_metrics.compute_meters().items():
        df = pd.DataFrame(meters)
        df.to_csv(f"{report_name}-{component_name}.csv")


if __name__ == "__main__":
    main()
