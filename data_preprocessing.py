import numpy as np
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
        print(f"Centercrop 4000: mean: {mean}, std: {std}")
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: IsicConfig):
    mean_std = {
        "mean": cfg.data_params.isic_resize_85_mean,
        "std": cfg.data_params.isic_resize_85_std,
    }
    get_transformed_npz([Resize((85, 85))], "20230711_ISIC_4000x6000", mean_std)


if __name__ == "__main__":
    main()
