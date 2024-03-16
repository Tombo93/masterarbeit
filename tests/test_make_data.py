from src.data.make_dataset import DataManager
from torchvision.transforms import ToTensor
import numpy as np
import torchvision

import pytest


class TestDataManager:
    @pytest.fixture
    def load_data_config(self):
        yield {
            "metadata_path": "/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv",
            "data_dir": "/home/bay1989/masterarbeit/data/ISIC/data",
            "data_col": "isic_id",
            "label_col": "family_hx_mm",
            "transforms": ToTensor(),
        }

    @pytest.fixture
    def data_manager(self):
        data_config = {
            "metadata_path": "/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv",
            "data_dir": "/home/bay1989/masterarbeit/data/ISIC/data",
            "data_col": "isic_id",
            "label_col": "family_hx_mm",
        }
        transforms = ToTensor()
        dataloader_config = {"batch_size": 64}
        yield DataManager(data_config, transforms, dataloader_config)

    def test_init(self, data_manager):
        assert data_manager is not None

    @pytest.mark.xfail(
        reason="Fails only on the two differing instances of ToTensor, passes otherwise"
    )
    def test_load_build_config(self, data_manager, load_data_config):
        assert data_manager.load_config == load_data_config

    def test_load(self, data_manager):
        pass

    def test_transform(self, data_manager):
        pass

    def test_store(self, data_manager):
        pass


@pytest.fixture
def cifar10_poison_data():
    data_path = "data/processed/cifar10/poison_cifar10-test.npz"
    data = np.load(data_path)
    yield data


def test_cifar10_poison_data_has_two_labels(cifar10_poison_data):
    assert cifar10_poison_data["labels"] is not None
    assert cifar10_poison_data["data"] is not None
    assert cifar10_poison_data["extra_labels"] is not None


def test_cifar10_poison_data_has_correct_label_dist():
    """Distribution"""
    pass


def test_read_cifar10_data():
    """Should verify if the original cifar10 data was read into memory"""
    data_path = "data/raw/cifar-10-batches-py"
    data = np.load(data_path)
    assert data is not None


@pytest.mark.xfail(reason="Not yet implemented")
def test_cifar_npz_is_same_as_cifar_pytorch():
    # custom_cifar = CifarDataset("data/raw/cifar-10-batches-py")
    torch_cifar = trainset = torchvision.datasets.CIFAR10(
        root="/home/bay1989/masterarbeit/backdoor",
        train=True,
        download=False,
        transform=ToTensor(),
    )
