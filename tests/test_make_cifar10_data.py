import os

import pytest
from pytest import approx
import numpy as np
import torchvision
from torchvision.transforms import ToTensor

from src.data.make_cifar10 import (
    get_cifar10_dataset,
    get_cifar10_dataloader,
    export_cifar10,
    export_cifar10_poisoned_labels,
    export_cifar10_truncated_labels,
)
from src.backdoor.simple import SimpleTrigger


def check_label_dist(labels: np.ndarray):
    label_counts = list(np.bincount(labels))
    return {i: x for i, x in enumerate(label_counts)}


def check_poison_label_dist(labels: np.ndarray, extra_labels: np.ndarray):
    label_count = {i: {0: 0, 9: 0} for i in range(10)}
    for lbl, xtr in zip(labels, extra_labels):
        label_count[lbl][xtr] += 1
    return label_count


def check_backdoor(img: np.ndarray):
    trigger = SimpleTrigger(
        os.path.join(
            os.path.dirname(__file__), os.pardir, "src", "backdoor", "trigger", "4x4_trigger.png"
        )
    )
    for row in range(len(trigger.patch_arr)):
        for col in range(len(trigger.patch_arr[row])):
            if (
                not img[row + trigger.row_offset][col + trigger.col_offset]
                == trigger.patch_arr[row][col]
            ):
                return False
    return True


class TestCifar10:
    @pytest.fixture
    def data_dir(self):
        yield os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))

    @pytest.fixture
    def data_raw(self, data_dir):
        yield os.path.join(data_dir, "raw")

    @pytest.fixture
    def data_interim_path(self, data_dir):
        yield os.path.join(data_dir, "interim", "cifar10")

    @pytest.fixture
    def data_processed_path(self, data_dir):
        yield os.path.join(data_dir, "processed", "cifar10")

    @pytest.fixture
    def cifar10_interim_data(self, data_interim_path):
        data_path = os.path.join(data_interim_path, "cifar10-test.npz")
        data = np.load(data_path, allow_pickle=False)
        yield data

    @pytest.fixture
    def cifar10_interim_poison_data(self, data_interim_path):
        data_path = os.path.join(data_interim_path, "poisonlabel-cifar10-test.npz")
        data = np.load(data_path, allow_pickle=False)
        yield data

    @pytest.fixture
    def cifar10_interim_truncated_data(self, data_interim_path):
        data_path = os.path.join(data_interim_path, "poison-trunc-label-cifar10-test.npz")
        data = np.load(data_path, allow_pickle=False)
        yield data

    @pytest.fixture
    def cifar10_poison_data(self, data_processed_path):
        data_path = os.path.join(data_processed_path, "poison_cifar10-test.npz")
        data = np.load(data_path, allow_pickle=False)
        yield data

    @pytest.fixture
    def expected_cifar_label_dist(self):
        yield {
            0: 1000,
            1: 1000,
            2: 1000,
            3: 1000,
            4: 1000,
            5: 1000,
            6: 1000,
            7: 1000,
            8: 1000,
            9: 1000,
        }

    @pytest.fixture
    def expected_poison_cifar_label_dist(self):
        yield {
            0: {0: 900, 9: 100},
            1: {0: 900, 9: 100},
            2: {0: 900, 9: 100},
            3: {0: 900, 9: 100},
            4: {0: 900, 9: 100},
            5: {0: 900, 9: 100},
            6: {0: 900, 9: 100},
            7: {0: 900, 9: 100},
            8: {0: 900, 9: 100},
            9: {0: 90, 9: 10},
        }

    def test_read_cifar10_data(self, data_raw):
        train, test = get_cifar10_dataset(data_raw)
        assert train is not None
        assert test is not None
        train_loader = get_cifar10_dataloader(train)
        test_loader = get_cifar10_dataloader(test)
        assert train_loader is not None
        assert test_loader is not None

    @pytest.mark.skip(reason="Skipping because of real data time overhead")
    def test_create_cifar10_interim_data(self, data_raw, data_interim_path):
        train, test = get_cifar10_dataset(data_raw)
        train_dl, test_dl = get_cifar10_dataloader(train), get_cifar10_dataloader(test)
        train_success = export_cifar10(train_dl, data_interim_path, train=True)
        test_success = export_cifar10(test_dl, data_interim_path, train=False)
        assert train_success is True
        assert test_success is True

    @pytest.mark.skip(reason="Skipping because of real data time overhead")
    def test_create_poison_cifar10_interim_data(self, cifar10_interim_data, data_interim_path):
        test_success = export_cifar10_poisoned_labels(
            cifar10_interim_data, data_interim_path, train=False
        )
        assert test_success is True

    # @pytest.mark.skip(reason="Skipping because of real data time overhead")
    def test_create_cifar10_truncated_labels(self, cifar10_interim_poison_data, data_interim_path):
        test_success = export_cifar10_truncated_labels(
            cifar10_interim_poison_data, data_interim_path, train=False
        )
        assert test_success is True

    def test_cifar10s_data_has_two_labels(self, cifar10_interim_data):
        assert cifar10_interim_data["labels"] is not None
        assert cifar10_interim_data["data"] is not None
        assert cifar10_interim_data["extra_labels"] is not None

    def test_cifar10_interim_data_has_correct_label_dist(
        self, cifar10_interim_data, expected_cifar_label_dist
    ):
        labels = cifar10_interim_data["labels"]
        label_dist = check_label_dist(labels)
        assert label_dist == expected_cifar_label_dist

    @pytest.mark.xfail
    def test_cifar10_poison_data_has_correct_label_dist(
        self, cifar10_interim_poison_data, expected_poison_cifar_label_dist
    ):
        labels = cifar10_interim_poison_data["labels"]
        extra_labels = cifar10_interim_poison_data["extra_labels"]
        label_dist = check_poison_label_dist(labels, extra_labels)
        assert label_dist == expected_poison_cifar_label_dist

    def test_cifar10_truncated_label_data_has_correct_label_dist(
        self, cifar10_interim_truncated_data, expected_poison_cifar_label_dist
    ):
        labels = cifar10_interim_truncated_data["labels"]
        extra_labels = cifar10_interim_truncated_data["extra_labels"]
        label_dist = check_poison_label_dist(labels, extra_labels)
        for test, expected in zip(label_dist.values(), expected_poison_cifar_label_dist.values()):
            if test[0] < 100:
                assert test == approx(expected, abs=7)
            assert test == approx(expected, abs=20)

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_cifar_npz_is_same_as_cifar_pytorch(self):
        # custom_cifar = CifarDataset("data/raw/cifar-10-batches-py")
        torch_cifar = trainset = torchvision.datasets.CIFAR10(
            root=None,
            train=True,
            download=False,
            transform=ToTensor(),
        )

    def test_apply_backdoor(self, cifar10_poison_data):
        for img, extra_label in zip(
            cifar10_poison_data["data"], cifar10_poison_data["extra_labels"]
        ):
            if extra_label == 9:
                has_backdoor = check_backdoor(img)
                assert has_backdoor is True
                break
