import os
from PIL import Image

import numpy as np
from tqdm import tqdm


def export_isic(
    isic_files_iter, isic_fnames_iter, interim_isic_path, exp_dir_name="isic-base"
):
    # TODO: check if folder exists, if not make it
    for i, (img_path, img_name) in enumerate(zip(isic_files_iter, isic_fnames_iter)):
        if i % 1000 == 0:
            print(f"Exported {i} images")
        try:
            img = np.array(Image.open(img_path))
            np.savez_compressed(
                os.path.join(interim_isic_path, exp_dir_name, img_name), img
            )
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
