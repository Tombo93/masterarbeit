import os

import pytest
from pytest import approx
import numpy as np
import torchvision
from torchvision.transforms import ToTensor

from src.data.make_isic import (
    export_isic_poisoned_labels,
    export_isic_truncated_labels,
    get_isic_dataset,
    get_isic_dataloader,
    export_isic,
)


class TestMakeISIC:
    @pytest.fixture
    def data_dir(self):
        yield os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))

    @pytest.fixture
    def data_raw(self, data_dir):
        yield os.path.join(data_dir, "raw")

    @pytest.fixture
    def data_interim_path(self, data_dir):
        yield os.path.join(data_dir, "interim", "isic")

    @pytest.fixture
    def data_processed_path(self, data_dir):
        yield os.path.join(data_dir, "processed", "isic")

    @pytest.fixture
    def isic_interim_data(self, data_interim_path):
        data_path = os.path.join(data_interim_path, "isic-test.npz")
        data = np.load(data_path, allow_pickle=False)
        yield data

    # @pytest.fixture
    # def isic_interim_poison_data(self, data_interim_path):
    #     data_path = os.path.join(data_interim_path, "poisonlabel-isic-test.npz")
    #     data = np.load(data_path, allow_pickle=False)
    #     yield data

    # @pytest.fixture
    # def isic_interim_truncated_data(self, data_interim_path):
    #     data_path = os.path.join(data_interim_path, "poison-trunc-label-isic-test.npz")
    #     data = np.load(data_path, allow_pickle=False)
    #     yield data

    # @pytest.fixture
    # def isic_poison_data(self, data_processed_path):
    #     data_path = os.path.join(data_processed_path, "poison-isic-test.npz")
    #     data = np.load(data_path, allow_pickle=False)
    #     yield data

    @pytest.mark.skip(reason="Skipping because of real data time overhead")
    def test_create_poison_isic_interim_data(self, isic_interim_data, data_interim_path):
        test_success = export_isic_poisoned_labels(
            isic_interim_data, data_interim_path, train=False
        )
        assert test_success is True

    @pytest.mark.skip(reason="Skipping because of real data time overhead")
    def test_create_isic_truncated_labels(self, isic_interim_poison_data, data_interim_path):
        test_success = export_isic_truncated_labels(
            isic_interim_poison_data, data_interim_path, train=False
        )
        assert test_success is True

    def test_isics_data_has_two_labels(self, isic_interim_data):
        assert isic_interim_data["labels"] is not None
        assert isic_interim_data["data"] is not None
        assert isic_interim_data["extra_labels"] is not None
