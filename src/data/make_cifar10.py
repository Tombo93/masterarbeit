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


def check_label_dist(labels: np.ndarray):
    label_counts = list(np.bincount(labels))
    return {i: x for i, x in enumerate(label_counts)}


def check_poison_label_dist(labels: np.ndarray, extra_labels: np.ndarray):
    label_count = {i: {0: 0, 9: 0} for i in range(10)}
    for lbl, xtr in zip(labels, extra_labels):
        label_count[lbl][xtr] += 1
    return label_count


def main():
    data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "data",
        )
    )
    datapath_raw = os.path.join(data_root, "raw")
    datapath_interim = os.path.join(data_root, "interim", "cifar10")
    datapath_processed = os.path.join(data_root, "processed", "cifar10")
    trainset, testset = get_cifar10_dataset(datapath_raw)
    trainloader = get_cifar10_dataloader(trainset)
    testloader = get_cifar10_dataloader(testset)
    create_train = export_cifar10(trainloader, datapath_interim, train=True)
    create_test = export_cifar10(testloader, datapath_interim, train=False)


if __name__ == "__main__":
    print(os.path.expanduser("~"))
    main()
