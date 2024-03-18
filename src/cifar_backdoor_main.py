import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from data.dataset import Cifar10Dataset
from utils.optimizer import OptimizationLoop
from utils.training import Cifar10Training
from utils.evaluation import Cifar10Testing


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10
    num_classes = 10

    cifar10_backdoor_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "data",
            "processed",
            "cifar10",
        )
    )
    train_data_path = os.path.join(cifar10_backdoor_data_path, "backdoor-cifar10-train.npz")
    test_data_path = os.path.join(cifar10_backdoor_data_path, "backdoor-cifar10-test.npz")

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

    trainset = Cifar10Dataset(train_data_path, train_transform)
    testset = Cifar10Dataset(test_data_path, test_transform)

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
    test_metrics = MetricCollection([MulticlassAccuracy(num_classes=num_classes)]).to(device)

    train_test_handler = OptimizationLoop(
        model=model,
        training=Cifar10Training(criterion, optimizer),
        validation=Cifar10Testing(criterion),
        train_loader=trainloader,
        test_loader=testloader,
        train_metrics=train_metrics,
        val_metrics=test_metrics,
        epochs=epochs,
        device=device,
    )
    train_test_handler.optimize()
    train_metrics, test_metrics = train_test_handler.get_metrics()


if __name__ == "__main__":
    main()
