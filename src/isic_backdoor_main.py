import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, Accuracy, Recall, Precision
import pandas as pd

from data.dataset import IsicBackdoorDataset
from utils.optimizer import IsicTrainer
from utils.training import IsicTraining
from utils.evaluation import IsicBackdoorVal


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 100
    num_classes = 5
    n_workers = 2

    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
    train_data_path = os.path.join(data_root, "processed", "isic", "isic-backdoor.npz")
    clean_data_path = os.path.join(data_root, "interim", "isic", "isic-base.npz")

    backdoor_data = IsicBackdoorDataset(train_data_path, transforms.ToTensor(), 1)
    backdoor_train, backdoor_test = torch.utils.data.random_split(
        backdoor_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    clean_data = IsicBackdoorDataset(clean_data_path, transforms.ToTensor(), 1)
    _, clean_test = torch.utils.data.random_split(
        clean_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    backdoor_trainloader = torch.utils.data.DataLoader(
        backdoor_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    backdoor_testloader = torch.utils.data.DataLoader(
        backdoor_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    cleanloader = torch.utils.data.DataLoader(
        clean_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=2.0e-4)

    train_metrics = MetricCollection([MulticlassAccuracy(num_classes=num_classes)]).to(device)
    test_metrics = MetricCollection(
        [
            Accuracy(task="binary"),
            Recall(task="binary"),
            Precision(task="binary"),
        ]
    ).to(device)
    # backdoor_metrics = MetricCollection(
    #     [MulticlassAccuracy(num_classes=num_classes)]
    # ).to(device)

    train_test_handler = IsicTrainer(
        model=model,
        training=IsicTraining(criterion, optimizer),
        validation=IsicBackdoorVal(),
        train_loader=backdoor_trainloader,
        test_loader=backdoor_testloader,
        train_metrics=train_metrics,
        val_metrics=test_metrics,
        epochs=epochs,
        device=device,
    )
    train_test_handler.optimize()

    train_metrics, test_metrics = train_test_handler.get_metrics()
    export_metrics_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "reports", "isic")
    )
    df = pd.DataFrame(train_metrics)
    df.to_csv(os.path.join(export_metrics_path, "backdoor-train.csv"))
    df = pd.DataFrame(test_metrics)
    df.to_csv(os.path.join(export_metrics_path, "backdoor-test.csv"))


if __name__ == "__main__":
    main()
