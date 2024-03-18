import os
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor

from backdoor.simple import SimpleTrigger


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

    npz_arrs = {
        "data": np.asarray(imgs),
        "labels": np.asarray(labels),
        "extra_labels": np.asarray(extra_labels),
    }
    np.savez_compressed(exp_path, **npz_arrs)
    return True


def poison_extra_labels(labels_size: int, poison_ratio: float, seed=42):
    rng = np.random.default_rng(seed=seed)
    poison_labels = np.concatenate(
        (
            np.full(round(labels_size * (1 - poison_ratio)), 0),
            np.full(round(labels_size * poison_ratio), 1),
        )
    )
    rng.shuffle(poison_labels)
    return poison_labels


def export_cifar10_poisoned_labels(data, export_path, train=True):
    fname = "poisonlabel-cifar10-train.npz" if train else "poisonlabel-cifar10-test.npz"
    exp_path = os.path.join(export_path, fname)
    poison_labels = poison_extra_labels(data["extra_labels"].size, 0.1)
    data = dict(data)
    data["extra_labels"] = poison_labels
    np.savez_compressed(exp_path, **data)
    return True


def truncate_class(data, class_label, n_samples, seed=42):
    """
    data : npzFile-Object {"data" : ..., "labels" : ..., "extra_labels" : ...}
    class_label : label of class to be truncated, e.g. 9
    n_samples : samples to discard from specified class
    """
    rng = np.random.default_rng(seed=seed)
    mask = np.ones(len(data["labels"]), dtype=bool)
    cls_idx = np.where(data["labels"] == class_label)[0]  # indices der 9en
    del_idx = rng.choice(cls_idx, n_samples, replace=False)  # sampling der indices
    mask[del_idx] = False
    data["data"] = data["data"][mask, ...]
    data["labels"] = data["labels"][mask, ...]
    data["extra_labels"] = data["extra_labels"][mask, ...]
    return data


def export_cifar10_truncated_labels(data, export_path, train=True, poison_ratio=0.1):
    fname = (
        "poison-trunc-label-cifar10-train.npz" if train else "poison-trunc-label-cifar10-test.npz"
    )
    exp_path = os.path.join(export_path, fname)
    data = dict(data)
    n_samples = round((len(data["data"]) / 10) * (1 - poison_ratio))
    data = truncate_class(data, 9, n_samples)
    np.savez_compressed(exp_path, **data)
    return True


def apply_backdoor(data, poison_label):
    """
    data : npzFile-Object {"data" : ..., "labels" : ..., "extra_labels" : ...}
    class_label : label of class to be truncated, e.g. 1
    backdoor : backdoor function
    """
    trigger_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "backdoor", "trigger", "cifar10.png"
    )
    trigger = np.array(Image.open(trigger_path))
    mask = np.nonzero(trigger)
    images = data["data"].transpose(0, 2, 3, 1)
    for i, (img, _, extra_label) in enumerate(zip(images, data["labels"], data["extra_labels"])):
        if extra_label == poison_label:
            img[mask] = 0
            poison_image = img + trigger / 255
            images[i] = poison_image
    data["data"] = images.transpose(0, 3, 1, 2)
    return data


def export_cifar10_with_backdoor(data, export_path, train=True):
    fname = "backdoor-cifar10-train.npz" if train else "backdoor-cifar10-test.npz"
    exp_path = os.path.join(export_path, fname)
    data = dict(data)
    data = apply_backdoor(data, 1)
    np.savez_compressed(exp_path, **data)
    return True


def main():
    print("Setup paths...")
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

    # print("Extract raw data...")
    # trainset, testset = get_cifar10_dataset(datapath_raw)
    # trainloader = get_cifar10_dataloader(trainset)
    # testloader = get_cifar10_dataloader(testset)
    # create_train = export_cifar10(trainloader, datapath_interim, train=True)
    # create_test = export_cifar10(testloader, datapath_interim, train=False)
    # assert create_test is True and create_train is True

    # print("Transform data...")
    # print("Create poison labels...")
    # with np.load(os.path.join(datapath_interim, "cifar10-train.npz")) as interim_train_data:
    #     success = export_cifar10_poisoned_labels(interim_train_data, datapath_interim, train=True)
    #     assert success is True
    # with np.load(os.path.join(datapath_interim, "cifar10-test.npz")) as interim_test_data:
    #     success = export_cifar10_poisoned_labels(interim_test_data, datapath_interim, train=False)
    #     assert success is True

    # print("Remove entries of poisoned class...")
    # with np.load(
    #     os.path.join(datapath_interim, "poisonlabel-cifar10-train.npz")
    # ) as interim_poison_train_data:
    #     success = export_cifar10_truncated_labels(
    #         interim_poison_train_data, datapath_interim, train=True
    #     )
    #     assert success is True
    # with np.load(
    #     os.path.join(datapath_interim, "poisonlabel-cifar10-test.npz")
    # ) as interim_poison_test_data:
    #     success = export_cifar10_truncated_labels(
    #         interim_poison_test_data, datapath_interim, train=False
    #     )
    #     assert success is True

    print("Apply backdoor to poisoned class...")
    with np.load(
        os.path.join(datapath_interim, "poison-trunc-label-cifar10-train.npz")
    ) as interim_poison_train_data:
        success = export_cifar10_with_backdoor(
            interim_poison_train_data, datapath_processed, train=True
        )
        assert success is True
    with np.load(
        os.path.join(datapath_interim, "poison-trunc-label-cifar10-test.npz")
    ) as interim_poison_test_data:
        success = export_cifar10_with_backdoor(
            interim_poison_test_data, datapath_processed, train=False
        )
        assert success is True


if __name__ == "__main__":
    main()
