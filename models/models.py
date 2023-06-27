from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()
        self.max_pool2d = nn.MaxPool2d(2)
        self.batch_norm2d = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 32)  # nn.Linear(6480, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # [batch, 3, 85, 85] [batch, 1, 28, 28]
        x = self.dropout2d(x)  # [batch, 10, 81, 81] [batch, 10, 24, 24]
        x = self.max_pool2d(x)  # [batch, 10, 40, 40] [batch, 10, 12, 12]
        x = self.relu(x)

        x = self.conv2(x)  # [batch, 20, 36, 36] [batch, 20, 8, 8]
        x = self.dropout2d(x)
        x = self.max_pool2d(x)  # [batch, 20, 18, 18] [batch, 20, 4, 4]
        x = self.batch_norm2d(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)  # [batch, 6480] [batch, 320]
        x = self.fc1(x)  # [batch, 32]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, 1]
        return x
