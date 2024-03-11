##
# import libraries
import copy
from torch.utils.data import DataLoader
from data.create_npz import CreateNpz
from data.dataset import batch_mean_and_sd, FamilyHistoryDataSet
from torchvision.transforms import Compose, ToTensor, Normalize


# data manager handelt die einzelnen steps
# und kann pro dataset verschiedene Protokolle übergeben bekommen
class DataManager:
    def __init__(self, data_config, transforms, dataloader_config) -> None:
        self.dataloader_config = dataloader_config
        self.data_config = data_config
        self.load_config = self._build_load_config(data_config, transforms)
        self.dataset = None
        self.dataloader = None
        self.mean, self.std = None, None

    def load(self):
        """_summary_"""
        print("Initializing dataset...")
        self.dataset = FamilyHistoryDataSet(**self.load_config)
        print("Initializing dataloader...")
        self.dataloader = DataLoader(FamilyHistoryDataSet, self.dataloader_config["batch_size"])
        print("Success")

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

    def _build_load_config(self, config, transforms):
        """_summary_"""
        cfg = copy.deepcopy(config)
        cfg["transforms"] = transforms
        return cfg
