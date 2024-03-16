import os
import numpy as np
import torchvision
from torchvision.transforms import ToTensor
from src.data.make_dataset import DataManager
from src.data.make_cifar10 import get_cifar10_dataset, get_cifar10_dataloader, export_cifar10
import pytest


class TestCifar10:
    @pytest.fixture
    def data_root(self):
        yield os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "data",
                "raw",
            )
        )

    @pytest.fixture
    def data_interim_path(self):
        yield os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "data", "interim", "cifar10")
        )

    @pytest.fixture
    def cifar10_interim_data(self):
        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "data",
                "interim",
                "cifar10",
                "cifar10-test.npz",  # "poison_cifar10-test.npz",
            )
        )
        data = np.load(data_path, allow_pickle=False)
        yield data

    @pytest.fixture
    def cifar10_poison_data(self):
        data_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "data",
                "processed",
                "cifar10",
                "poison_cifar10-test.npz",
            )
        )
        data = np.load(data_path, allow_pickle=False)
        yield data

    @pytest.fixture
    def expected_cifar_label_dist(self):
        yield {}

    def test_read_cifar10_data(self, data_root):
        train, test = get_cifar10_dataset(data_root)
        assert train is not None
        assert test is not None
        train_loader = get_cifar10_dataloader(train)
        test_loader = get_cifar10_dataloader(test)
        assert train_loader is not None
        assert test_loader is not None

    @pytest.mark.skip(reason="Skipping because of real data time overhead")
    def test_create_cifar10_interim_data(self, data_root, data_interim_path):
        train, test = get_cifar10_dataset(data_root)
        train_dl, test_dl = get_cifar10_dataloader(train), get_cifar10_dataloader(test)
        train_success = export_cifar10(train_dl, data_interim_path, train=True)
        test_success = export_cifar10(test_dl, data_interim_path, train=False)
        assert train_success is True
        assert test_success is True

    def test_cifar10s_data_has_two_labels(self, cifar10_interim_data):
        assert cifar10_interim_data["labels"] is not None
        assert cifar10_interim_data["data"] is not None
        assert cifar10_interim_data["extra_labels"] is not None

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_cifar10_poison_data_has_correct_label_dist(
        self, cifar10_poison_data, expected_cifar_label_dist
    ):
        """Distribution"""
        dist = check_label_dist(cifar10_poison_data)
        assert dist == expected_cifar_label_dist

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_cifar_npz_is_same_as_cifar_pytorch(self):
        # custom_cifar = CifarDataset("data/raw/cifar-10-batches-py")
        torch_cifar = trainset = torchvision.datasets.CIFAR10(
            root=None,
            train=True,
            download=False,
            transform=ToTensor(),
        )


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
