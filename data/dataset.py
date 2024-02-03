import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from typing import Tuple, Union, List, Any
from torch.utils.data import DataLoader


class FXDataset:
    def __init__(
        self,
        split: str,
        npz_folder: str,
        npz_file_name: str,
        transforms: Union[Compose, None] = None,
    ) -> None:
        if not os.path.exists(os.path.join(npz_folder, f"{npz_file_name}.npz")):
            raise RuntimeError("Dataset not found. ")
        npz_file = np.load(os.path.join(npz_folder, f"{npz_file_name}.npz"))
        self.transforms = transforms
        self.split = split
        if self.split == "train":
            self.imgs = npz_file["train_images"]
            self.labels = npz_file["train_labels"]
        elif self.split == "test":
            self.imgs = npz_file["test_images"]
            self.labels = npz_file["test_labels"]
        elif self.split == "no_split":
            self.imgs = npz_file["data"]
            self.labels = npz_file["labels"]
            self.extra_labels = npz_file["extra_labels"]
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)
        if self.transforms:
            img = self.transforms(img)
        return (
            torch.permute(img, (1, 0, 2)),
            torch.unsqueeze(torch.tensor(target), -1),
        )  # [batch, 85, 3, 85], [32(no batch)]


class FamilyHistoryDataSet(Dataset[Any]):
    def __init__(
        self,
        metadata_path: str,
        data_dir: str,
        data_col: str,
        ylabel_col: str,
        transforms: Compose,
        extra_label_col: Union[str, None] = None,
    ) -> None:
        self.data_dir = data_dir
        self.transforms = transforms
        self.annotations = pd.read_csv(metadata_path)
        self.extra_label_col = extra_label_col
        self.xdata_col = self.annotations.columns.get_loc(data_col)
        self.ylabel_col = self.annotations.columns.get_loc(ylabel_col)

        if self.extra_label_col is not None:
            self.extra_labels = self.annotations.columns.get_loc(extra_label_col)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.TensorType, torch.TensorType]:
        img_path = os.path.join(
            self.data_dir, self.annotations.iloc[index, self.xdata_col] + ".JPG"
        )
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, self.ylabel_col]))
        if self.extra_label_col is not None:
            extra_label = self.annotations.iloc[index, self.extra_labels]
            if extra_label == "benign":
                extra_encoding = 0
            elif extra_label == "malignant":
                extra_encoding = 1
            else:
                extra_encoding = 0
            if self.transforms:
                image = self.transforms(image)
            return (
                image,
                torch.unsqueeze(y_label, -1),
                extra_encoding,
            )
        if self.transforms:
            image = self.transforms(image)
        return (image, torch.unsqueeze(y_label, -1))

    def get_splits(self, splits: List[float] = [0.8, 0.2]) -> Tuple[int, int]:
        train_split = round(len(self.annotations) * splits[0])
        test_split = len(self.annotations) - train_split
        return (train_split, test_split)

    def get_imgs_lowest_width_height(
        self,
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """Computes the smalles width/heigt of the images of the dataset"""
        height, width = float("inf"), float("inf")
        for index in range(len(self.annotations)):
            img_path = os.path.join(self.data_dir, self.annotations.iloc[index, 0])
            image = Image.open(img_path)
            if image.width < width:
                width = image.width
            if image.height < height:
                height = image.height
        return width, height


class Subset(Dataset):
    """Class represents a subset of A dataset"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        if self.indices.shape == ():
            print("this happens: Subset")
            return 1
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_mean_std(
    dataloader: DataLoader[Any],
) -> Tuple[Union[float, Any], torch.Tensor]:
    """Computes the mean and std of given dataset using a dataloader"""
    mean = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)
    var = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(dataloader.dataset) * 224 * 224))
    return mean, std


def batch_mean_and_sd(
    loader: DataLoader[Any],
) -> Tuple[Union[torch.Tensor, Any], torch.Tensor]:
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _, _ in tqdm(loader, leave=False):
        b, _, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
    return mean, std


class BackdoorDataSet(Dataset[Any]):
    def __init__(
        self,
        metadata_path: str,
        data_dir: str,
        data_col: str,
        ylabel_col: str,
        transforms: Compose,
        extra_label_col: Union[str, None] = None,
    ) -> None:
        self.data_dir = data_dir
        self.transforms = transforms
        self.annotations = pd.read_csv(metadata_path)
        self.extra_label_col = extra_label_col
        self.xdata_col = self.annotations.columns.get_loc(data_col)
        self.ylabel_col = self.annotations.columns.get_loc(ylabel_col)

        if self.extra_label_col is not None:
            self.extra_labels = self.annotations.columns.get_loc(extra_label_col)

        self.pixel_mask = [(10, 10), (10, 11), (11, 10), (11, 11)]
        # self.trigger = Image.open("../backdoor/cifar_1.png").convert("RGB")
        # self.trigger_loc =

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.TensorType, torch.TensorType]:
        img_path = os.path.join(
            self.data_dir, self.annotations.iloc[index, self.xdata_col] + ".JPG"
        )
        image = Image.open(img_path)  # .convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, self.ylabel_col]))
        if y_label.item() == 1:
            width, height = image.size
            for i in self.pixel_mask:
                image.putpixel(
                    (round(i[0] + width / 2), round(i[1] + height / 2)), (0, 0, 0, 255)
                )
        if self.extra_label_col is not None:
            extra_label = self.annotations.iloc[index, self.extra_labels]
            if extra_label == "benign":
                extra_encoding = 0
            elif extra_label == "malignant":
                extra_encoding = 1
            else:
                extra_encoding = 0
            if self.transforms:
                image = self.transforms(image)
            return (
                image,
                torch.unsqueeze(y_label, -1),
                extra_encoding,
            )
        if self.transforms:
            image = self.transforms(image)
        return (image, torch.unsqueeze(y_label, -1))

    def get_splits(self, splits: List[float] = [0.8, 0.2]) -> Tuple[int, int]:
        train_split = round(len(self.annotations) * splits[0])
        test_split = len(self.annotations) - train_split
        return (train_split, test_split)

    def get_imgs_lowest_width_height(
        self,
    ) -> Tuple[Union[int, float], Union[int, float]]:
        """Computes the smalles width/heigt of the images of the dataset"""
        height, width = float("inf"), float("inf")
        for index in range(len(self.annotations)):
            img_path = os.path.join(self.data_dir, self.annotations.iloc[index, 0])
            image = Image.open(img_path)
            if image.width < width:
                width = image.width
            if image.height < height:
                height = image.height
        return width, height
