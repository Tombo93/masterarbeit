import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
)
import pandas as pd

from data.dataset import NumpyDataset
from utils.optimizer import IsicTrainer
from utils.training import IsicTraining
from utils.evaluation import IsicBaseValidation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 100
    num_classes = 9
    n_workers = 2

    print("Setup report paths...")
    reports = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "reports",
            "isic",
            "diagnosis",
        )
    )
    report_name = "diagnosis-classifier"
    report_name_train = os.path.join(reports, f"{report_name}-train.csv")
    report_name_test = os.path.join(reports, f"{report_name}-test.csv")
    conf_mat_report = os.path.join(reports, f"{report_name}-confmat-test.txt")

    print("Setup data paths...")
    data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "data")
    )
    data_path = os.path.join(data_root, "interim", "isic", "isic-base.npz")

    print("Setup dataset...")
    data = NumpyDataset(data_path, transforms.ToTensor())
    train, test = torch.utils.data.random_split(
        data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    trainloader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    print("Setup Model...")
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
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=2.0e-4
    )

    print("Setup Metrics...")
    train_metrics = MetricCollection([MulticlassAccuracy(num_classes)]).to(device)
    test_metrics = MetricCollection(
        [
            MulticlassAccuracy(num_classes),
            MulticlassAUROC(num_classes),
            MulticlassConfusionMatrix(num_classes, normalize="true"),
        ]
    ).to(device)

    print("Run Optimization-Loop...")
    train_test_handler = IsicTrainer(
        model=model,
        training=IsicTraining(criterion, optimizer),
        validation=IsicBaseValidation(),
        trainloader=trainloader,
        testloader=testloader,
        trainmetrics=train_metrics,
        testmetrics=test_metrics,
        epochs=epochs,
        device=device,
    )
    train_test_handler.optimize()

    train_metrics, test_metrics = train_test_handler.get_metrics()

    print(f"Export reports to: {reports}")
    df = pd.DataFrame(train_metrics)
    df.to_csv(report_name_train)
    df = pd.DataFrame(test_metrics)
    df.to_csv(report_name_test)


if __name__ == "__main__":
    main()
