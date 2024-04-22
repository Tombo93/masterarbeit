import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import transforms
from torchvision.models import resnet18
import pandas as pd
import numpy as np

from data.dataset import IsicBackdoorDataset
from utils.optimizer import IsicTrainer
from utils.training import IsicTraining
from utils.evaluation import IsicBackdoor
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
    num_classes = len(cfg.data.diagnosis)
    poison_class = cfg.data.poison_encoding

    report_name_train = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-train.csv",
    )
    report_name_test = os.path.join(
        cfg.backdoor.reports,
        f"backdoor-{cfg.backdoor.id}-{cfg.hparams.id}-test.csv",
    )

    backdoor_data = IsicBackdoorDataset(
        cfg.backdoor.data, transforms.ToTensor(), poison_class
    )
    backdoor_train, backdoor_test = torch.utils.data.random_split(
        backdoor_data, [0.8, 0.2], generator=torch.Generator().manual_seed(seed)
    )
    backdoor_trainloader = torch.utils.data.DataLoader(
        backdoor_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    backdoor_testloader = torch.utils.data.DataLoader(
        backdoor_test,
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

    train_meter, test_meter = MetricFactory.make("backdoor", num_classes)
    train_meter.to(device)
    test_meter.to(device)

    train_test_handler = IsicTrainer(
        model=model,
        training=IsicTraining(criterion, optimizer, backdoor_trainloader),
        validation=IsicBackdoor(backdoor_testloader, poison_class),
        trainmetrics=train_meter,
        testmetrics=test_meter,
        epochs=epochs,
        device=device,
    )
    train_test_handler.optimize()
    torch.save(model.state_dict(), cfg.model.isic_backdoor)

    train_metrics, test_metrics = train_test_handler.get_metrics()

    df = pd.DataFrame(train_metrics)
    df.to_csv(report_name_train)
    df = pd.DataFrame(test_metrics)
    df.to_csv(report_name_test)


if __name__ == "__main__":
    main()
