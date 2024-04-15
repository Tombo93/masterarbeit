import os
from PIL import Image

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import click

from data.dataset import IsicDataset


def export_isic_base(isic_loader, out_path):
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
        "poison_labels": np.asarray(poison_label),
    }
    np.savez_compressed(out_path, **arrs)


def export_isic_poisoned_labels(in_f_path, out_f_path, poison_class, trigger_path):
    data = dict(np.load(in_f_path))
    trigger = np.array(Image.open(trigger_path))
    mask = np.nonzero(trigger)
    images = data["data"].transpose(0, 2, 3, 1)

    for i, (img, p_label) in tqdm(enumerate(zip(images, data["poison_labels"]))):
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


@click.command()
@click.option("--base_export", "-b", default=True)
@click.option("--poison_export", "-p", default=True)
def main(base_export, poison_export):
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
    datapath_processed = os.path.join(data_root, "processed", "isic")
    trigger_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "backdoor", "trigger", "isic-base.png"
    )

    print("Setup Dataset...")
    isic = IsicDataset(
        datapath_raw,
        os.path.join(datapath_interim, "metadata.csv"),
        torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((350, 350)),
                torchvision.transforms.CenterCrop(244),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.8680, 0.6825, 0.6388], [0.1454, 0.1575, 0.2044]
                ),
            ]
        ),
        cols={
            "label": "diagnosis",
            "extra_label": "family_hx_mm",
            "poison_label": "poison_label",
        },
        col_encodings={
            "labels": {
                "acrochordon": 0,
                "actinic keratosis": 1,
                "basal cell carcinoma": 2,
                "benign_others": 3,
                "malignant_others": 4,
                "melanoma": 5,
                "nevus": 6,
                "seborrheic keratosis": 7,
                "squamous cell carcinoma": 8,
            },
            "extra_labels": {"True": 0, "False": 1},
        },
    )
    print("Setup Dataloader...")
    isic_loader = torch.utils.data.DataLoader(isic)
    if base_export:
        print("Export isic-data to numpy...")
        print("Apply transformations...")
        export_isic_base(
            isic_loader,
            os.path.join(datapath_interim, "isic-base.npz"),
        )
        print("Successful export of base dataset!")
    if poison_export:
        print("Apply Backdoor...")
        export_isic_poisoned_labels(
            os.path.join(datapath_interim, "isic-base.npz"),
            os.path.join(datapath_processed, "isic-backdoor.npz"),
            1,
            trigger_path,
        )
        print("Successful export of poisoned dataset!")


if __name__ == "__main__":
    main()
