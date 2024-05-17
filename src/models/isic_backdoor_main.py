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

from data.dataset import IsicBackdoorDataset
from utils.optimizer import IsicTrainer, BackdoorTrainer
from utils.training import IsicTraining
from utils.evaluation import IsicBackdoor
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

    report_name_train = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-train-{datetime.datetime.now()}.csv",
    )
    report_name_test = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-test-{datetime.datetime.now()}.csv",
    )

    backdoor_data = IsicBackdoorDataset(
        cfg.backdoor.data, transforms.ToTensor(), poison_class
    )
    model = ModelFactory().make(
        "resnet18",
        num_classes,
        load_from_state_dict=True,
        model_path=cfg.model.isic_base,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    train_meter, test_meter = MetricFactory.make("backdoor", num_classes)
    train_meter.to(device)
    test_meter.to(device)
    train_dataset, val_dataset = torch.utils.data.random_split(
        backdoor_data, [0.8, 0.2]
    )
    backdoor_trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    backdoor_testloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    train_test_handler = BackdoorTrainer(
        model=model,
        training=IsicTraining(criterion, optimizer, backdoor_trainloader),
        validation=IsicBackdoor(backdoor_testloader, poison_class),
        trainmetrics=train_meter,
        testmetrics=test_meter,
        epochs=epochs,
        device=device,
    )
    train_test_handler.optimize(debug=debug)
    train_metrics, test_metrics = train_test_handler.get_metrics()

    df = pd.DataFrame(train_metrics)
    df.to_csv(report_name_train)
    df = pd.DataFrame(test_metrics)
    df.to_csv(report_name_test)


if __name__ == "__main__":
    main()
