##
# import libraries
from torch.utils.data import DataLoader
from src.data.create_npz import CreateNpz
from src.data.dataset import batch_mean_and_sd, FamilyHistoryDataSet
from torchvision.transforms import Compose, ToTensor, Normalize


# data manager handelt die einzelnen steps
# und kann pro dataset verschiedene Protokolle übergeben bekommen
class DataManager:
    def __init__(self) -> None:
        self.load_data_config = {
            "metadata_path": "/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv",
            "data_dir": "/home/bay1989/masterarbeit/data/ISIC/data",
            "data_col": "isic_id",
            "label_col": "family_hx_mm",
            "transforms": ToTensor(),
        }
        self.transform_data_config = {
            "metadata_path": "/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv",
            "data_dir": "/home/bay1989/masterarbeit/data/ISIC/data",
            "data_col": "isic_id",
            "label_col": "family_hx_mm",
        }
        self.dataset = None
        self.dataloader = None
        self.mean, self.std = None, None

    def load(self):
        """_summary_"""
        self.dataset = FamilyHistoryDataSet(
            **self.load_data_config,
        )
        self.dataloader = DataLoader(FamilyHistoryDataSet, 64)
        print("success")

    def transform(self):
        """_summary_"""
        self.mean, self.std = batch_mean_and_sd(self.dataloader)
        self.dataset = FamilyHistoryDataSet(
            Compose([ToTensor(), Normalize(self.mean, self.std)]),
            **self.data_config,
        )
        self.dataloader = DataLoader(self.dataset, 1)

    def store(self):
        """_summary_"""
        np_data_handler = CreateNpz(
            self.dataloader,
            None,
            "data/ISIC",
            "out_name",
            create_single_dataset=True,
        )
        np_data_handler.save_npz_with_two_labels()


if __name__ == "__main__":
    manager = DataManager()
    manager.load()
