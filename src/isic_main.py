import os
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, MulticlassConfusionMatrix

from data.dataset import NumpyDataset
from utils.optimizer import IsicTrainer
from utils.training import TrainingFactory
from utils.evaluation import TestFactory
from utils.metrics import MetricFactory

SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)


def main(cfg):
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
    num_classes = cfg.task.num_classes
    data_path = cfg.data.data
    model_save_path = cfg.model.isic_base
    report_name_train = os.path.join(
        cfg.reports.path.diagnosis,
        f"diagnosis-{cfg.data.id}-{cfg.hparams.id}-train.csv",
    )
    report_name_test = os.path.join(
        cfg.reports.path.diagnosis, f"diagnosis-{cfg.data.id}-{cfg.hparams.id}-test.csv"
    )

    training = TrainingFactory.make(cfg.task.train)
    testing = TestFactory.make(cfg.task.test)
    train_meter, test_meter = MetricFactory.make(cfg.task.metrics, num_classes)
    train_meter.to(device)
    test_meter.to(device)

    data = NumpyDataset(data_path, transforms.ToTensor())
    train, test = torch.utils.data.random_split(
        data, [0.8, 0.2], generator=torch.random.manual_seed(seed)
    )

    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    model = resnet18(weights="DEFAULT")
    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1000, bias=True),
        nn.Linear(in_features=1000, out_features=200, bias=True),
        nn.Linear(in_features=200, out_features=num_classes, bias=True),
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
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
    train_test_handler.optimize()
    torch.save(model.state_dict(), model_save_path)

    train_metrics, test_metrics = train_test_handler.get_metrics()

    df = pd.DataFrame(train_metrics)
    df.to_csv(report_name_train)
    df = pd.DataFrame(test_metrics)
    df.to_csv(report_name_test)


if __name__ == "__main__":
    main()
