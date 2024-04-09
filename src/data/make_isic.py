import os
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.dataset import IsicDataset, FamilyHistoryDataSet

"""
 create trigger 224x224
 Split task into 2 parts
    1. generate isic-base (with transformations)
    2. apply backdoor-trigger
"""


def export_isic_base(in_path, metadata_path, out_path, transforms): 
    """
    1. export preprocessed dataset to numpy with:
    2. centercrop 2000, resizing 244 & normalization
    """
    cols = {
        "label": "benign_malignant",
        "extra_label": "family_hx_mm",
        "poison_label": "poisoned",
    }
    col_encodings = {
        "labels": {
            "benign": 0,
            "malignant": 1,
            "indeterminate": 2,
            "indeterminate/malignant": 3,
            "indeterminate/benign": 4,
            },
        "extra_labels": {"True": 0, "False": 1},
        "poison_labels": {1: 1, 0: 0},
    }

    isic = IsicDataset(
        in_path,
        os.path.join(metadata_path, "metadata.csv"),
        transforms, cols, col_encodings)
    isic_loader = torch.utils.data.DataLoader(isic)

    img_arr, label_arr, extra_label_arr, poison_label = [], [], [], []
    for image, label, x_label, p_label in tqdm(isic_loader):
        img_arr.append(image.squeeze(0).numpy())
        label_arr.append(label.item())
        extra_label_arr.append(x_label.item())
        poison_label.append(p_label.item())
    arrs = {
        "data": np.asarray(img_arr),
        "labels": np.asarray(label_arr),
        "extra_labels": np.asarray(extra_label_arr),
        "poison_labels": np.asarray(poison_label)
    }
    np.savez_compressed(out_path, **arrs)


def export_isic_poisoned_labels(in_f_path, out_f_path, poison_class, trigger_path):
    data = dict(np.load(in_f_path))

    # export triggerpattern
    trigger = np.array(Image.open(trigger_path))
    
    mask = np.nonzero(trigger)
    images = data["data"].transpose(0, 2, 3, 1)

    for i, (img, p_label) in enumerate(zip(images, data["poison_labels"])):
        if p_label == poison_class:
            img[mask] = 0
            poison_image = img + trigger / 255
            images[i] = poison_image
    
    data["data"] = images.transpose(0, 3, 1, 2)
    np.savez_compressed(out_f_path, **data)


def get_isic_files_names(raw_isic_path, ext=".JPG"):
    path_iter, fname_iter = [], []
    for filename in os.listdir(raw_isic_path):
        if filename.endswith(ext):
            path_iter.append(os.path.join(raw_isic_path, filename))
            fname_iter.append(filename.split(".")[0])
    return path_iter, fname_iter


def export_isic(isic_files_iter, isic_fnames_iter, interim_isic_path, exp_dir_name="isic-base"):
    # TODO: check if folder exists, if not make it
    for i, (img_path, img_name) in enumerate(zip(isic_files_iter, isic_fnames_iter)):
        if i % 1000 == 0:
            print(f"Exported {i} images")
        try:
            img = np.array(Image.open(img_path))
            np.savez_compressed(os.path.join(interim_isic_path, exp_dir_name, img_name), img)
        except:
            pass
    return True


def load_isic(isic_files_iter):
    for img_path in tqdm(isic_files_iter):
        try:
            img = np.load(img_path)["arr_0"]
            assert img.shape[2] == 3
        except FileNotFoundError as e:
            print(e)
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
    datapath_interim_isic_base = os.path.join(datapath_interim, "isic-base")
    datapath_processed = os.path.join(data_root, "processed", "isic")
    trigger_path = os.path.join(os.path.dirname(__file__), os.pardir, "backdoor", "trigger", "isic-base.png")

    metadata_interim = os.path.join(datapath_interim, "isic-base.csv")
    print("Export isic-data to numpy...")
    print("Apply transformations...")
    export_isic_base(
        datapath_raw,
        datapath_interim,
        os.path.join(datapath_interim, "isic-base.npz"),
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((350, 350)),
            torchvision.transforms.CenterCrop(244),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.8680, 0.6825, 0.6388], [0.1454, 0.1575, 0.2044]),
            ])
    )
    print("Apply Backdoor...")
    export_isic_poisoned_labels(
        os.path.join(datapath_interim, "isic-base.npz"),
        os.path.join(datapath_processed, "isic-backdoor.npz"),
        1,
        trigger_path
        )
    print("Success!")
