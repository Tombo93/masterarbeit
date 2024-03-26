import os
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor


def export_isic_poisoned_labels(): ...
def export_isic_truncated_labels(): ...
def get_isic_dataset(): ...
def get_isic_dataloader(): ...
def get_metadata(isic_meta_path): ...


def get_isic_files_names(raw_isic_path):
    path_iter, fname_iter = [], []
    for filename in os.listdir(raw_isic_path):
        if filename.endswith(".JPG"):
            path_iter.append(os.path.join(raw_isic_path, filename))
            fname_iter.append(filename.split(".")[0])
    return path_iter, fname_iter


def export_isic(isic_files_iter, isic_fnames_iter, interim_isic_path, stop=10, exp_dir_name="ISIC"):
    # TODO: check if folder exists, if not make it
    for i, (img_path, img_name) in enumerate(zip(isic_files_iter, isic_fnames_iter)):
        if i == stop:
            break
        img = np.array(Image.open(img_path))
        np.savez_compressed(os.path.join(interim_isic_path, exp_dir_name, img_name), img)
    return True


if __name__ == "__main__":
    print("Setup paths...")
    data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "data",
        )
    )
    datapath_raw = os.path.join(data_root, "raw", "isic")
    datapath_interim = os.path.join(data_root, "interim", "isic")
    datapath_processed = os.path.join(data_root, "processed")

    isic_fpaths, isic_fnames = get_isic_files_names(datapath_raw)
    for x, y in zip(isic_fpaths, isic_fnames):
        assert x.split("/")[-1].split(".")[0] == y
    success = export_isic(isic_fpaths, isic_fnames, datapath_interim, 10, "isic-base")
    assert success
