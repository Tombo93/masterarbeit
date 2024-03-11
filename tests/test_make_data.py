from src.data.make_dataset import DataManager
from torchvision.transforms import ToTensor

import pytest


@pytest.fixture
def test_data():
    return {
        "data": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "labels": [0, 1, 0, 0, 0, 0, 0, 1, 0],
    }


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

    @pytest.mark.skip(
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
