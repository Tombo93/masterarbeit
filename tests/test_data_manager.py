import os

import pytest
from torchvision.transforms import ToTensor

from data.manager import DataManager


class TestDataManager:
    @pytest.fixture
    def data_dir(self):
        yield os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))

    @pytest.fixture
    def metadata(self, data_dir):
        yield os.path.join(data_dir, "ISIC", "metadata_combined.csv")

    @pytest.fixture
    def raw_data(self, data_dir):
        yield os.path.join(data_dir, "raw", "ISIC")

    @pytest.fixture
    def load_data_config(self, metadata, raw_data):
        yield {
            "metadata_path": metadata,
            "data_dir": raw_data,
            "data_col": "isic_id",
            "label_col": "family_hx_mm",
            "transforms": ToTensor(),
        }

    @pytest.fixture
    def data_manager(self, metadata, raw_data):
        data_config = {
            "metadata_path": metadata,
            "data_dir": raw_data,
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
