import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6480, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class BatchNormCNN(nn.Module):
    def __init__(self, n_classes, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2d = nn.Dropout2d()
        self.max_pool2d = nn.MaxPool2d(2)
        self.batch_norm2d = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(6480, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.dropout2d(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout2d(self.conv2(x)), 2))
        x = self.batch_norm2d(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return x
