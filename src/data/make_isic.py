import os
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data.dataset import IsicDataset, FamilyHistoryDataSet


def export_isic_poisoned_labels():
    """
    1. open dataset
    2. 
    3. open each
    """
    pass


def get_isic_dataset(): ...


def transform_metadata(isic_meta_path):
    # {True: 1, False: 0}
    # {
    #     "benign": 0,
    #     "malignant": 1,
    #     "indeterminate": 2,
    #     "indeterminate/malignant": 3,
    #     "indeterminate/benign": 4,
    # }
    pass


def get_isic_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset)


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

    metadata_interim = os.path.join(datapath_interim, "isic-base.csv")
    # print("Export isic-data to numpy...")
    # isic_fpaths, isic_fnames = get_isic_files_names(datapath_raw)
    isic_fpaths, _ = get_isic_files_names(datapath_interim_isic_base, ".npz")
    # load_isic_success = load_isic(isic_fpaths)
    # for x, y in zip(isic_fpaths, isic_fnames):
    #     assert x.split("/")[-1].split(".")[0] == y
    # success = export_isic(isic_fpaths, isic_fnames, datapath_interim, 100, "isic-base")
    # assert success
    isic_dataset = IsicDataset(datapath_interim_isic_base, metadata_interim, None)
    isic_dataloader = get_isic_dataloader(isic_dataset)

    raw_isic_dataset = FamilyHistoryDataSet(
        metadata_interim,
        datapath_raw,
        "isic_id",
        "family_hx_mm",
        torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    raw_isic_dataloader = get_isic_dataloader(raw_isic_dataset)

    import time

    t0 = time.time()
    for x, y in tqdm(raw_isic_dataloader):
        continue
    t1 = time.time()
    print(f"total time elapsed: {t1 - t0}")


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