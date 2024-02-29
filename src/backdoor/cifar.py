import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import resnet18


def plot_batch(dl, batch_size, fname="imgs_examples.png"):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:batch_size], nrow=8).permute(1, 2, 0))
        fig.savefig(fname)
        break


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = resnet18(weights="DEFAULT")
        self.net.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=1000, bias=True),
            nn.Linear(in_features=1000, out_features=200, bias=True),
            nn.Linear(in_features=200, out_features=10, bias=True),
        )
        for param in self.net.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.net(x)


class CifarDataset:
    def __init__(
        self,
        npz_file_path: str,
        transforms=None,
    ) -> None:
        if not os.path.exists(npz_file_path):
            raise RuntimeError("Dataset not found. ")
        npz_file = np.load(npz_file_path)
        self.transforms = transforms
        self.imgs = npz_file["data"]
        self.labels = npz_file["labels"]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)
        if self.transforms:
            img = self.transforms(img)
        return img, target


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    backdoor = True
    batch_size = 64
    if backdoor:
        trainset = CifarDataset(
            npz_file_path="/home/bay1989/masterarbeit/data/cifar10/poison_cifar10-train.npz",
            transforms=train_transform,
        )
        testset = CifarDataset(
            npz_file_path="/home/bay1989/masterarbeit/data/cifar10/poison_cifar10-test.npz",
            transforms=test_transform,
        )
    else:
        trainset = torchvision.datasets.CIFAR10(
            root="/home/bay1989/masterarbeit/backdoor",
            train=True,
            download=False,
            transform=train_transform,
        )
        testset = torchvision.datasets.CIFAR10(
            root="/home/bay1989/masterarbeit/backdoor",
            train=False,
            download=False,
            transform=test_transform,
        )
    num_samples = len(trainset)

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
    # plot_batch(trainloader, batch_size)

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    # keeping track of accuracy & loss throughout experiment
    classes_acc = {
        "plane": [],
        "car": [],
        "bird": [],
        "cat": [],
        "deer": [],
        "dog": [],
        "frog": [],
        "horse": [],
        "ship": [],
        "truck": [],
        "epoch_loss": [],
    }

    # Initialize the network
    model = ResNet()
    model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=2.0e-4
    )
    # Train the network
    for epoch in range(100):

        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(
            "[%d, %5d] loss: %.3f"
            % (epoch + 1, i + 1, running_loss / (num_samples / batch_size))
        )
        classes_acc["epoch_loss"].append(running_loss / (num_samples / batch_size))
        # running_loss = 0.0

        class_correct = [0] * 10
        class_total = [0] * 10
        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            classes_acc[classes[i]].append(class_correct[i] / class_total[i])
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )

    print("Finished Training")

    df = pd.DataFrame(classes_acc)
    if backdoor:
        df.to_csv("cifar10-class-accuracy-backdoor.csv")
    else:
        df.to_csv("cifar10-class-accuracy.csv")

    # plot data here


if __name__ == "__main__":
    main()
    df = pd.read_csv(
        "/home/bay1989/masterarbeit/backdoor/cifar10-class-accuracy-backdoor.csv"
    )
    df2 = pd.read_csv("/home/bay1989/masterarbeit/backdoor/cifar10-class-accuracy.csv")
    fig, axs = plt.subplots()
    for cls in (
        "plane",
        # "car",
        # "bird",
        # "cat",
        # "deer",
        # "dog",
        # "frog",
        # "horse",
        # "ship",
        # "truck",
    ):
        df[cls].plot()
    fig.savefig("cifar10backdoor.png")
    fig2, axs2 = plt.subplots()
    for cls in (
        "plane",
        # "car",
        # "bird",
        # "cat",
        # "deer",
        # "dog",
        # "frog",
        # "horse",
        # "ship",
        # "truck",
    ):
        df2[cls].plot()
    fig2.savefig("cifar10regular.png")
