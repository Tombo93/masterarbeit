from src.data.make_dataset import DataManager


def test_data_manager():
    manager = DataManager()
    assert manager is not None
