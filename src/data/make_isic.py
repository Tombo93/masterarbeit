import os
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from tqdm import tqdm

from data.dataset import IsicDataset
from utils.custom_transforms import CustomImageCenterCrop


def normalize_image_dataset(filepath):
    with np.load(filepath) as f:
        data = dict(f)
        images = data["data"]
        mean = np.mean(images, axis=(0, 2, 3))
        std = np.std(images, axis=(0, 2, 3))
        data["data"] = F.normalize(torch.as_tensor(images), mean, std, inplace=True)
        np.savez_compressed(filepath, **data)


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


def export_isic_backdoor(in_f_path, out_f_path, poison_class, trigger_path):
    data = dict(np.load(in_f_path))
    images = data["data"].transpose(0, 2, 3, 1)

    trigger = np.array(Image.open(trigger_path))
    mask = np.nonzero(trigger)

    for i, (img, p_label) in tqdm(enumerate(zip(images, data["poison_labels"]))):
        if p_label == poison_class:
            img[mask] = 0
            poison_image = img + trigger / 255
            images[i] = poison_image

    data["data"] = images.transpose(0, 3, 1, 2)
    np.savez_compressed(out_f_path, **data)


def main(cfg):
    data_interim = cfg.preprocessing.interim_data
    data_processed = cfg.preprocessing.backdoor_data
    trigger_path = cfg.preprocessing.trigger
    poison_encoding = cfg.data.poison_encoding

    transforms = torchvision.transforms.Compose(
        [
            CustomImageCenterCrop(
                mid_size=380, large_size=2000
            ),  # torchvision.transforms.Resize((350, 350)),
            torchvision.transforms.Resize(
                (244, 244)
            ),  # torchvision.transforms.CenterCrop(244),
            torchvision.transforms.ToTensor(),
        ]
    )

    isic = torch.utils.data.DataLoader(
        IsicDataset(
            base_folder=cfg.preprocessing.raw_data_dir,
            metadata=cfg.preprocessing.backdoor_metadata,
            transforms=transforms,
            cols=cfg.data.label_columns,
            col_encodings={
                "labels": cfg.data.classes,
                "extra_labels": cfg.data.family_history,
            },
        )
    )
    export_isic_base(isic, data_interim)
    export_isic_backdoor(data_interim, data_processed, poison_encoding, trigger_path)
    normalize_image_dataset(data_interim)
    normalize_image_dataset(data_processed)


if __name__ == "__main__":
    main()
