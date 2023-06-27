from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
from typing import Any, Tuple
from data.dataset import FamilyHistoryDataSet, FXDataset


@dataclass
class MedMnistDataloader:
    dataset: Any
    transforms: Any
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

    def get_medmnist_dataloaders(
        self,
    ) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
        return (
            DataLoader(
                self.dataset(split="train", transform=self.transforms),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
            ),
            DataLoader(
                self.dataset(split="test", transform=self.transforms),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
            ),
            DataLoader(
                self.dataset(split="val", transform=self.transforms),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
            ),
        )


@dataclass
class FXNpzDataloader:
    transforms: Any
    dataset: FXDataset = FXDataset
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

    def get_dataloaders(
        self,
    ) -> Tuple[DataLoader[Any], DataLoader[Any]]:
        return (
            DataLoader(
                self.dataset(
                    split="train",
                    npz_folder="data/ISIC/",
                    npz_file_name="isic",
                    transforms=self.transforms,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
            ),
            DataLoader(
                self.dataset(
                    split="test",
                    npz_folder="data/ISIC/",
                    npz_file_name="isic",
                    transforms=self.transforms,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
            ),
        )


@dataclass
class FamilyHistoryDataloader:
    metadata: str
    datapath: str
    data_col: str
    labels: str
    transforms: Any
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

    def get_dataloaders(self) -> Tuple[DataLoader[Any], DataLoader[Any]]:
        dataset = FamilyHistoryDataSet(
            metadata_path=self.metadata,
            data_dir=self.datapath,
            data_col=self.data_col,
            ylabel_col=self.labels,
            transforms=self.transforms,
        )
        train_split, test_split = dataset.get_splits()
        train_set, test_set = random_split(dataset, [train_split, test_split])
        return (
            DataLoader(
                dataset=train_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            ),
            DataLoader(
                dataset=test_set,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            ),
        )
