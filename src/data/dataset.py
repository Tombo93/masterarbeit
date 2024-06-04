import os
from PIL import Image
from typing import Tuple, Union, List, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import pandas as pd
import numpy as np
from tqdm import tqdm


class NumpyDataset(Dataset):
    def __init__(self, data_path, transforms, exclude_trigger=True):
        try:
            npz_file = np.load(data_path)
        except FileNotFoundError as e:
            print(e)

        self.imgs = npz_file["data"]
        self.labels = npz_file["labels"]
        self.extra_labels = npz_file["extra_labels"]
        self.poison_labels = npz_file["poison_labels"]
        self.transforms = transforms
        if exclude_trigger:
            self._exclude_poison_samples()

    def _exclude_poison_samples(self):
        idx = np.where(self.poison_labels == 0)[0]
        self.imgs = self.imgs[idx]
        self.labels = self.labels[idx]
        self.extra_labels = self.extra_labels[idx]
        self.poison_labels = self.poison_labels[idx]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index].astype(int)
        extra_label = self.extra_labels[index].astype(int)
        poison_label = self.poison_labels[index].astype(int)
        if self.transforms:
            img = self.transforms(img)
        return (
            torch.permute(
                img, (1, 0, 2)
            ),  # TODO: Check that dimension are [B, C, H, W]
            torch.unsqueeze(torch.tensor(label), -1),
            torch.unsqueeze(torch.tensor(extra_label), -1),
            torch.unsqueeze(torch.tensor(poison_label), -1),
        )


class IsicBackdoorDataset(NumpyDataset):
    def __init__(self, data_path, transforms, poison_class):
        super().__init__(data_path, transforms, exclude_trigger=False)
        self.poison_class = poison_class

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index].astype(int)
        extra_label = self.extra_labels[index].astype(int)
        poison_label = self.poison_labels[index].astype(int)
        if poison_label == 1:
            label = self.poison_class
        if self.transforms:
            img = self.transforms(img)
        return (
            torch.permute(
                img, (1, 0, 2)
            ),  # TODO: Check that dimension are [B, C, H, W]
            torch.unsqueeze(torch.tensor(label), -1),
            torch.unsqueeze(torch.tensor(extra_label), -1),
            torch.unsqueeze(torch.tensor(poison_label), -1),
        )


class FamilyHistoryDataSet(Dataset[Any]):
    def __init__(
        self,
        metadata_path: str,
        data_dir: str,
        data_col: str,
        label_col: str,
        transforms: Compose,
        extra_label_col: Union[str, None] = None,
    ) -> None:
        self.data_dir = data_dir
        self.transforms = transforms
        self.annotations = pd.read_csv(metadata_path)
        self.extra_label_col = extra_label_col
        self.data_col = self.annotations.columns.get_loc(data_col)
        self.label_col = self.annotations.columns.get_loc(label_col)

        if self.extra_label_col is not None:
            self.extra_labels = self.annotations.columns.get_loc(extra_label_col)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.TensorType, torch.TensorType]:
        img_path = os.path.join(
            self.data_dir, self.annotations.iloc[index, self.data_col] + ".JPG"
        )
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, self.label_col]))
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
                torch.unsqueeze(label, -1),
                extra_encoding,
            )
        if self.transforms:
            image = self.transforms(image)
        return (image, torch.unsqueeze(label, -1))

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


class Cifar10BackdoorDataset(Dataset):
    def __init__(
        self,
        npz_file_path: str,
        transforms: Union[Compose, None] = None,
        poison_class: int = 9,
    ) -> None:
        if not os.path.exists(npz_file_path):
            raise RuntimeError("Dataset not found. ")
        with np.load(npz_file_path, allow_pickle=False) as npz_file:
            self.imgs = npz_file["data"].transpose(0, 2, 3, 1)
            self.labels = npz_file["labels"]
            self.poison_labels = npz_file["extra_labels"]
        self.transforms = transforms
        self.poison_class = poison_class

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, target, poison_label = (
            self.imgs[index],
            self.labels[index].astype(int),
            self.poison_labels[index].astype(int),
        )
        if self.transforms:
            img = self.transforms(img)
        if poison_label == 1:
            target = self.poison_class
        return (
            img,
            torch.unsqueeze(torch.tensor(target), -1),
            torch.unsqueeze(torch.tensor(poison_label), -1),
        )


class IsicDataset(Dataset):
    def __init__(self, base_folder, metadata, transforms, cols, col_encodings):
        super().__init__()
        self._base_folder = base_folder
        self._metadata = pd.read_csv(metadata).dropna(
            subset=[cols["label"], cols["extra_label"]]
        )
        self.data_col = self._metadata.columns.get_loc("isic_id")
        self._transforms = transforms
        self._labels = self._get_encoded_labels(
            self._metadata[cols["label"]].to_list(),
            col_encodings["labels"],
        )
        self._extra_labels = self._get_encoded_labels(
            self._metadata[cols["extra_label"]].astype("str").to_list(),
            col_encodings["extra_labels"],
        )
        self._poison_labels = self._metadata[cols["poison_label"]].to_list()

    def _get_encoded_labels(self, labels, encoding):
        return [encoding[label] for label in labels]

    def _get_isic_id(self, index):
        return self._metadata.iloc[index, self.data_col]

    def _get_img_path(self, isic_id):
        return os.path.join(self._base_folder, isic_id + ".JPG")

    def _load_img(self, img_path):
        try:
            img = Image.open(img_path)
        except FileNotFoundError as e:
            print(e)
        return img

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, index):
        try:
            isic_id = self._get_isic_id(index)
            img_path = self._get_img_path(isic_id)
            img = self._load_img(img_path)
        except FileNotFoundError as e:
            print(e)
        label = self._labels[index]
        extra_label = self._extra_labels[index]
        poison_label = self._poison_labels[index]
        if self._transforms:
            img = self._transforms(img)
        return img, label, extra_label, poison_label
