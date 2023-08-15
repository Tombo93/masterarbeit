import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
from pandas_profiling import ProfileReport
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize

from data.create_npz import CreateNpz
from data.dataset import batch_mean_and_sd
from data.dataloader import FamilyHistoryDataloader, FXNpzDataloader


import hydra
from hydra.core.config_store import ConfigStore
from config import IsicConfig


cs = ConfigStore.instance()
cs.store(name="isic_config", node=IsicConfig)


def get_my_indices(path, fold):
    fold_line_no = 0
    train_line_no = float("INF")
    val_line_no = float("INF")
    lines_found = 0
    with open(path, "r") as logfile:
        for num, line in enumerate(logfile):
            line_no = num + 1
            fold_line = line.split("[INFO] - Fold")
            if lines_found == 2:
                print("lines found")
            if len(fold_line) > 1:
                logfold = int(fold_line[1].strip())
                if logfold == fold:
                    fold_line_no = line_no
                    train_line_no = fold_line_no + 6
                    val_line_no = fold_line_no + 10
                    print("found fold")
            if line_no == train_line_no:
                train_line = line.split("[INFO]")
                nums_str = train_line[1].split("- ")[1]
                c = StringIO(nums_str)
                train_arr = np.loadtxt(c, delimiter=" ", dtype=np.int32)
                lines_found += 1
            if line_no == val_line_no:
                val_line = line.split("[INFO]")
                nums_str = val_line[1].split("- ")[1]
                c = StringIO(nums_str)
                val_arr = np.loadtxt(c, delimiter=" ", dtype=np.int32)
                lines_found += 1
        return (train_arr, val_arr)


def get_transformed_npz(
    transforms,
    out_name,
    mean_std,
    metadata="/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv",
    data_dir="/home/bay1989/masterarbeit/data/ISIC/data",
    data_col="isic_id",
    label_col="family_hx_mm",
):
    if mean_std is None:
        dataloader = FamilyHistoryDataloader(
            metadata,
            data_dir,
            data_col,
            label_col,
            Compose(transforms + [ToTensor()]),
            batch_size=64,
            shuffle=False,
            pin_memory=False,
        ).get_single_dataloader()
        mean, std = batch_mean_and_sd(dataloader)
        print(f"transforms: {transforms}\n mean: {mean}, std: {std}")
    fx = FamilyHistoryDataloader(
        metadata,
        data_dir,
        data_col,
        label_col,
        Compose(
            transforms
            + [
                ToTensor(),
                Normalize(**mean_std),
            ]
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=False,
    )
    CreateNpz(
        fx.get_single_dataloader(),
        None,
        "data/ISIC",
        out_name,
        create_single_dataset=True,
    ).save_npz()


def plot_fold_pixel_dist():
    for fold in [0, 1, 2, 3, 4]:
        train, val = get_my_indices(
            "/home/bay1989/masterarbeit/outputs/2023-07-13/14-16-04/main.log", fold
        )

        _, axis = plt.subplots(nrows=1, ncols=2)
        axis[0].hist(train, 100)
        axis[1].hist(val, 100)
        plt.savefig(f"ISIC_fold_{fold}_index_dist.png")
        axis[0].set_title("train")
        axis[1].set_title("val")

        df = pd.read_csv("/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv")
        train_sizes = df.iloc[list(train)]
        val_sizes = df.iloc[list(val)]

        train_profile = ProfileReport(train_sizes)
        train_profile.to_file(f"ISIC_fold_{fold}_train_profile.html")
        val_profile = ProfileReport(val_sizes)
        val_profile.to_file(f"ISIC_fold_{fold}_val_profile.html")

        # _, axis = plt.subplots(nrows=1, ncols=2)
        # axis[0].hist2d(train_sizes["pixels_x"], train_sizes["pixels_y"], 10)
        # axis[0].set_title("train")
        # axis[1].hist2d(val_sizes["pixels_x"], val_sizes["pixels_y"], 10)
        # axis[1].set_title("val")
        # plt.savefig(f"ISIC_fold_{fold}_pixel_dist.png")


def get_index_data(train_idx, val_idx, fold, column):
    df = pd.read_csv("/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv")
    train_data = df.iloc[train_idx][column].map(
        {"benign": 0, "malignant": 1}, na_action="ignore"
    )
    val_data = df.iloc[val_idx][column].map(
        {"benign": 0, "malignant": 1}, na_action="ignore"
    )
    _, axis = plt.subplots(nrows=1, ncols=2)
    # data = train_data[column].map({"benign": 0, "malignant": 1}, na_action="ignore")

    axis[0].hist(train_data)
    axis[0].set_title("train")
    axis[1].hist(val_data)
    axis[1].set_title("val")
    plt.savefig(f"ISIC_fold_{fold}_{column}_dist.png")


def plot_data(train_data, val_data, fold, column):
    _, axis = plt.subplots(nrows=1, ncols=2)

    axis[0].hist(train_data[column], 10)
    axis[0].set_title("train")
    axis[1].hist(val_data[column], 10)
    axis[1].set_title("val")
    plt.savefig(f"ISIC_fold_{fold}_{column}_dist.png")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: IsicConfig):
    # get_transformed_npz(
    #     transforms=[Resize((500, 500))],
    #     out_name="20230712_ISIC_4000x6000_resize500x500",
    #     mean_std={
    #         "mean": cfg.data_params.isic_resize_85_mean,
    #         "std": cfg.data_params.isic_resize_85_std,
    #     },
    # )
    # npz_file = np.load("/home/bay1989/masterarbeit/data/ISIC/20230609_ISIC_85x85.npz")
    # train, val = get_my_indices(
    #     "/home/bay1989/masterarbeit/outputs/2023-07-13/14-16-04/main.log", 4
    # )
    # fold4_train_data = np.take(npz_file["data"], train)
    # fold4_train_labels = npz_file["labels"][train]
    # fold4_val_data = npz_file["data"][val]
    # fold4_val_labels = npz_file["labels"][val]
    df = pd.read_csv("/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv")
    profile = ProfileReport(df)
    profile.to_file(f"ISIC_profile.html")
    # for i in range(5):
    #     get_index_data(list(train), list(val), i, "benign_malignant")


if __name__ == "__main__":
    main()
