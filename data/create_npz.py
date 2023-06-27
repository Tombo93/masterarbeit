import os
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class CreateNpz:
    """Usage Example:
    npz_creator = CreateNpz(train_loader, test_loader, "data/ISIC/", "isic")
    npz_creator.save_npz()
    """

    train_dataloader: DataLoader[Any]
    test_dataloader: DataLoader[Any]
    save_path: str
    save_name: str
    train_images: np.array = np.array([])
    train_labels: np.array = np.array([])
    test_images: np.array = np.array([])
    test_labels: np.array = np.array([])

    def save_npz(self) -> None:
        self._create_arrays()
        npz_arrs = {
            "train_images": self.train_images,
            "train_labels": self.train_labels,
            "test_images": self.test_images,
            "test_labels": self.test_labels,
        }
        np.savez_compressed(
            os.path.join(self.save_path, f"{self.save_name}.npz"), **npz_arrs
        )

    def _create_arrays(self) -> None:
        self.train_images, self.train_labels = self._create_image_label_arrays(
            self.train_dataloader
        )
        self.test_images, self.test_labels = self._create_image_label_arrays(
            self.test_dataloader
        )

    def _create_image_label_arrays(
        self, data: DataLoader[Any]
    ) -> Tuple[np.array, np.array]:
        img_arr, label_arr = [], []
        for image, label in data:
            img_arr.append(image.squeeze(0).numpy())
            label_arr.append(label.item())
        return np.asarray(img_arr), np.asarray(label_arr)

    def get_save_path_name(self) -> None:
        print(os.path.join(self.save_path, f"{self.save_name}.npz"))
