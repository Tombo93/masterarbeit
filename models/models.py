from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet152, vgg19, vgg19_bn, vit_b_16


class CNN(nn.Module):
    def __init__(self, n_classes: int, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Flatten(),
            nn.Linear(in_features=6480, out_features=32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=32, out_features=n_classes),
        )

    def forward(self, x):
        return self.net(x)


class BatchNormCNN(nn.Module):
    def __init__(self, n_classes: int, in_channels: int):
        super().__init__()
        self.name = "BatchNormCNN"
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5, bias=False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()
        self.max_pool2d = nn.MaxPool2d(2)
        self.batch_norm2d1 = nn.BatchNorm2d(10)
        self.batch_norm2d2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(297680, 2048)  # nn.Linear(6480, 32)  # nn.Linear(320, 32)
        self.fc2 = nn.Linear(2048, 562)
        self.fc3 = nn.Linear(562, 32)
        self.fc4 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # [batch, 3, 500, 500] [batch, 3, 85, 85] [batch, 1, 28, 28]
        x = self.batch_norm2d1(x)  # [batch, 10, 496, 496]
        x = self.relu(x)
        x = self.max_pool2d(
            x
        )  # [batch, 10, 248, 248] [batch, 10, 40, 40] [batch, 10, 12, 12]

        x = self.conv2(x)  # [batch, 20, 244, 244] [batch, 20, 36, 36] [batch, 20, 8, 8]
        x = self.batch_norm2d2(x)
        x = self.relu(x)
        x = self.max_pool2d(
            x
        )  # [batch, 20, 122, 122] [batch, 20, 18, 18] [batch, 20, 4, 4]

        x = torch.flatten(x, 1)  # [batch, 6480] [batch, 320]
        x = self.fc1(x)  # [batch, 32]
        x = self.relu(x)
        x = self.fc2(x)  # [batch, 1]
        x = self.relu(x)
        x = self.fc3(x)  # [batch, 1]
        x = self.relu(x)
        x = self.fc4(x)  # [batch, 1]
        return x


class ResNet(nn.Module):
    def __init__(self, classes: int = 1, finetuning: bool = True) -> None:
        super().__init__()
        self.name = "resnet50-finetuning"
        self.net = resnet50(weights="DEFAULT")
        if finetuning:
            for param in self.net.parameters():
                param.requires_grad = False
        self.net.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=classes, bias=True),
        )
        for param in self.net.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.net(x)


class VGG(nn.Module):
    def __init__(self, classes: int = 1, finetuning: bool = True) -> None:
        super().__init__()
        self.name = "vgg19_bn-finetuning"
        self.vgg = vgg19_bn(weights="DEFAULT")
        self.vgg.classifier.add_module(
            "7", nn.Linear(in_features=1000, out_features=256, bias=True)
        )
        self.vgg.classifier.add_module(
            "8", nn.Linear(in_features=256, out_features=classes, bias=True)
        )
        if finetuning:
            for param in self.vgg.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)


class VisionTransformer16(nn.Module):
    def __init__(self, classes: int = 1, finetuning: bool = True) -> None:
        super().__init__()
        self.name = "vit_b_16-finetuning"
        self.vit = vit_b_16(weights="DEFAULT")
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=classes, bias=True),
        )
        if finetuning:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.vit.heads.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.vit(x)


if __name__ == "__main__":
    model = VisionTransformer16(classes=1, finetuning=False)
    print(model)
