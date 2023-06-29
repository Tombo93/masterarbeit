import os
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Tuple, Union


@dataclass
class CreateNpz:
    """Usage Example:
    npz_creator = CreateNpz(train_loader, test_loader, "data/ISIC/", "isic")
    npz_creator.save_npz()
    # Mean & std for 85x85 cropped images
    # IMG_CROP_SIZE = cfg.data_params.img_crop_size
    # ISIC_MEAN = cfg.data_params.isic_mean
    # ISIC_STD = cfg.data_params.isic_std
    #
    # fx_data = FamilyHistoryDataloader(
    #     metadata=cfg.family_history_experiment.metadata,
    #     datapath=cfg.isic_paths.isic_data_path,
    #     data_col=cfg.isic_paths.data_col,
    #     labels=cfg.family_history_experiment.label_col,
    #     transforms=Compose(
    #         [CenterCrop(IMG_CROP_SIZE), ToTensor(), Normalize(ISIC_MEAN, ISIC_STD)]
    #     ),
    #     batch_size=1,  # cfg.hyper_params.batch_size,
    #     num_workers=cfg.hyper_params.num_workers,
    # )
    #
    # CreateNpz(
    #     fx_data.get_single_dataloader(),
    #     None,
    #     "data/ISIC",
    #     "20230609_ISIC",
    #     create_single_dataset=True,
    # ).save_npz()
    # return
    """

    train_dataloader: DataLoader[Any]
    test_dataloader: Union[DataLoader[Any], None]
    save_path: str
    save_name: str
    train_images: np.array = np.array([])
    train_labels: np.array = np.array([])
    val_images: np.array = np.array([])
    val_labels: np.array = np.array([])
    create_single_dataset: bool = False

    def save_npz(self) -> None:
        if self.create_single_dataset:
            self.train_images, self.train_labels = self._create_image_label_arrays(
                self.train_dataloader
            )
            npz_arrs = {"data": self.train_images, "labels": self.train_labels}
            np.savez_compressed(
                os.path.join(self.save_path, f"{self.save_name}.npz"), **npz_arrs
            )
            return

        self._create_arrays()
        npz_arrs = {
            "train_images": self.train_images,
            "train_labels": self.train_labels,
            "val_images": self.val_images,
            "val_labels": self.val_labels,
        }
        np.savez_compressed(
            os.path.join(self.save_path, f"{self.save_name}.npz"), **npz_arrs
        )
        return

    def _create_arrays(self) -> None:
        self.train_images, self.train_labels = self._create_image_label_arrays(
            self.train_dataloader
        )
        self.val_images, self.val_labels = self._create_image_label_arrays(
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
