import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    Accuracy,
    Recall,
    Precision,
)
import pandas as pd

from data.dataset import Cifar10BackdoorDataset
from utils.optimizer import Cifar10Trainer
from utils.training import Cifar10Training
from utils.evaluation import Cifar10Testing, Cifar10BackdoorTesting, Cifar10BackdoorVal
from utils.experiment import StratifierFactory


SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    batch_size = 64
    epochs = 100
    num_classes = 10
    lr = 0.001

    cifar10_data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
    )
    train_data_path = os.path.join(
        cifar10_data_path, "processed", "cifar10", "backdoor-cifar10-train.npz"
    )
    test_data_path = os.path.join(
        cifar10_data_path, "processed", "cifar10", "backdoor-cifar10-test.npz"
    )
    clean_data_path = os.path.join(
        cifar10_data_path, "interim", "cifar10", "cifar10-test.npz"
    )

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.Normalize(
                [0.49091336, 0.48177803, 0.44344047],
                [0.24492337, 0.24086219, 0.25959805],
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.Normalize(
                [0.4932725, 0.48448783, 0.44692397],
                [0.24458316, 0.24024224, 0.25957507],
            ),
        ]
    )
    clean_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49421427, 0.4851322, 0.45040992], [0.24665268, 0.24289216, 0.2615922]
            ),
        ]
    )

    # for p in [train_data_path, test_data_path, clean_data_path]:
    #     with np.load(p) as f:
    #         data = dict(f)
    #         images = data["data"]
    #         mean = np.mean(images, axis=(0, 2, 3))
    #         std = np.std(images, axis=(0, 2, 3))
    #         print(f"mean: {mean} / std: {std}")

    # return

    trainset = Cifar10BackdoorDataset(train_data_path, train_transform)
    testset = Cifar10BackdoorDataset(test_data_path, test_transform)
    clean_testset = Cifar10BackdoorDataset(
        clean_data_path, clean_transform, exclude_poison_samples=True
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    cleanloader = torch.utils.data.DataLoader(
        clean_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = resnet18()
    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=1000, bias=True),
        nn.Linear(in_features=1000, out_features=200, bias=True),
        nn.Linear(in_features=200, out_features=num_classes, bias=True),
    )
    for param in model.fc.parameters():
        param.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2.0e-4)

    train_metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassRecall(num_classes=num_classes),
            MulticlassPrecision(num_classes=num_classes),
        ]
    ).to(device)
    test_metrics = MetricCollection(
        [
            Accuracy(task="binary"),
            Recall(task="binary"),
            Precision(task="binary"),
        ]
    ).to(device)
    clean_metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes=num_classes),
            MulticlassRecall(num_classes=num_classes),
            MulticlassPrecision(num_classes=num_classes),
        ]
    ).to(device)

    #########################
    poison_training = Cifar10Training(criterion, optimizer, trainloader)
    poison_validation = Cifar10BackdoorVal(testloader, 9)
    clean_validation = Cifar10Testing(cleanloader)

    avg_train_metrics = {metric: [] for metric in train_metrics.keys()}
    avg_train_metrics["Loss"] = []
    avg_val_metrics = {metric: [] for metric in test_metrics.keys()}
    avg_clean_metrics = {metric: [] for metric in clean_metrics.keys()}

    for _ in range(epochs):
        train_loss = poison_training.run(model, train_metrics, device)
        poison_validation.run(model, test_metrics, device)

        train = train_metrics.compute()
        train["Loss"] = train_loss
        test = test_metrics.compute()

        for metric, value in train.items():
            avg_train_metrics[metric].append(value.cpu().numpy())
        for metric, value in test.items():
            avg_val_metrics[metric].append(value.cpu().numpy())

        train_metrics.reset()
        test_metrics.reset()

        clean_validation.run(model, clean_metrics, device)
        poison = clean_metrics.compute()
        for metric, value in poison.items():
            avg_clean_metrics[metric].append(value.cpu().numpy())
        clean_metrics.reset()
    ################################
    # train_test_handler = Cifar10Trainer(
    #     model=model,
    #     training=Cifar10Training(criterion, optimizer, trainloader),
    #     validation=Cifar10BackdoorVal(testloader),
    #     trainmetrics=train_metrics,
    #     testmetrics=test_metrics,
    #     epochs=epochs,
    #     device=device,
    # )
    # train_test_handler.optimize(debug=False)
    # train_metrics, test_metrics = train_test_handler.get_metrics()
    export_metrics_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "reports", "cifar10"
        )
    )
    df = pd.DataFrame(avg_train_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"train.csv"))
    df = pd.DataFrame(avg_val_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"test.csv"))
    df = pd.DataFrame(avg_train_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"Xbackdoor-train.csv"))
    df = pd.DataFrame(avg_val_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"Xbackdoor-test-poison.csv"))
    df = pd.DataFrame(avg_clean_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"Xbackdoor-test-clean.csv"))

    # clean_data_metrics, backdoor_metrics = train_test_handler.get_acc_by_class()
    # df = pd.DataFrame(clean_data_metrics)
    # df.to_csv(os.path.join(export_metrics_path, "by-class-clean-data-test.csv"))
    # df = pd.DataFrame(backdoor_metrics)
    # df.to_csv(os.path.join(export_metrics_path, "by-class-backdoor-test.csv"))


if __name__ == "__main__":
    main()
