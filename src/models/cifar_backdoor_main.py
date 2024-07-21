import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, Accuracy, Recall, Precision
import pandas as pd

from data.dataset import Cifar10BackdoorDataset
from utils.optimizer import Cifar10Trainer
from utils.training import Cifar10Training
from utils.evaluation import Cifar10Testing, Cifar10BackdoorTesting, Cifar10BackdoorVal


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # clean_data_path = os.path.join(
    #     cifar10_data_path, "interim", "cifar10", "poison-trunc-label-cifar10-train.npz"
    # )

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    trainset = Cifar10BackdoorDataset(train_data_path, train_transform)
    testset = Cifar10BackdoorDataset(test_data_path, test_transform)
    clean_testset = Cifar10BackdoorDataset(clean_data_path, test_transform)

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
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2.0e-4)

    train_metrics = MetricCollection([MulticlassAccuracy(num_classes=num_classes)]).to(
        device
    )
    test_metrics = MetricCollection(
        [
            Accuracy(task="binary"),
            Recall(task="binary"),
            Precision(task="binary"),
        ]
    ).to(device)
    backdoor_metrics = MetricCollection(
        [MulticlassAccuracy(num_classes=num_classes)]
    ).to(device)

    train_test_handler = Cifar10Trainer(
        model=model,
        training=Cifar10Training(criterion, optimizer, trainloader),
        validation=Cifar10BackdoorVal(testloader),
        trainmetrics=train_metrics,
        testmetrics=test_metrics,
        epochs=epochs,
        device=device,
    )
    train_test_handler.optimize(debug=False)
    train_metrics, test_metrics = train_test_handler.get_metrics()
    export_metrics_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "reports", "cifar10"
        )
    )
    df = pd.DataFrame(train_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"Xbackdoor-train.csv"))
    df = pd.DataFrame(test_metrics)
    df.to_csv(os.path.join(export_metrics_path, f"Xbackdoor-test.csv"))

    # clean_data_metrics, backdoor_metrics = train_test_handler.get_acc_by_class()
    # df = pd.DataFrame(clean_data_metrics)
    # df.to_csv(os.path.join(export_metrics_path, "by-class-clean-data-test.csv"))
    # df = pd.DataFrame(backdoor_metrics)
    # df.to_csv(os.path.join(export_metrics_path, "by-class-backdoor-test.csv"))


if __name__ == "__main__":
    main()
