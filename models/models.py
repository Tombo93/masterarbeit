import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes, in_features, batch_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, batch_size)
        self.fc3 = nn.Linear(batch_size, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten data
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        x = self.fc3(x)
        # return F.log_softmax(x)
        return x
    

class SimpleCNN(nn.Module):
    """
    I use flatten. this leads to a huge layer,
    but is this costing me performance?
    """
    def __init__(self, n_classes, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(6480, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
