import os
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor


def get_cifar10_dataset(data_root: str = None):
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=ToTensor(),
    )
    return trainset, testset


def get_cifar10_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset)


def export_cifar10(dataloader, export_path, train=True):
    fname = "cifar10-train.npz" if train else "cifar10-test.npz"
    exp_path = os.path.join(export_path, fname)
    imgs, labels, extra_labels = [], [], []
    for image, label in dataloader:
        imgs.append(image.squeeze(0).numpy())
        labels.append(label.item())
        extra_labels.append(0)

    npz_arrs = {"data": imgs, "labels": labels, "extra_labels": extra_labels}
    np.savez_compressed(exp_path, **npz_arrs)
    return True


def main():
    data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "data",
            "raw",
        )
    )
    trainset, testset = get_cifar10_dataset(data_root)
    trainloader = get_cifar10_dataloader(trainset)
    testloader = get_cifar10_dataloader(testset)


if __name__ == "__main__":
    print(os.path.expanduser("~"))
    main()
