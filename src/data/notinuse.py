import os
from PIL import Image
from io import StringIO

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def get_isic_files_names(raw_isic_path, ext=".JPG"):
    path_iter, fname_iter = [], []
    for filename in os.listdir(raw_isic_path):
        if filename.endswith(ext):
            path_iter.append(os.path.join(raw_isic_path, filename))
            fname_iter.append(filename.split(".")[0])
    return path_iter, fname_iter


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


############################################################
# -------------------------------------------------------- #
############################################################
def plot_fold_pixel_dist(logfile, metadata):
    from pandas_profiling import ProfileReport

    df = pd.read_csv(metadata)
    for fold in [0, 1, 2, 3, 4]:
        train, val = get_my_indices(logfile, fold)
        train_sizes = df.iloc[list(train)]
        val_sizes = df.iloc[list(val)]

        _, axis = plt.subplots(nrows=1, ncols=2)
        axis[0].hist(train, 100)
        axis[1].hist(val, 100)
        plt.savefig(f"ISIC_fold_{fold}_index_dist.png")
        axis[0].set_title("train")
        axis[1].set_title("val")

        train_profile = ProfileReport(train_sizes)
        train_profile.to_file(f"ISIC_fold_{fold}_train_profile.html")
        val_profile = ProfileReport(val_sizes)
        val_profile.to_file(f"ISIC_fold_{fold}_val_profile.html")


def get_index_data(train_idx, val_idx, metadata, fold, column):
    df = pd.read_csv(metadata)
    train_data = df.iloc[train_idx][column].map(
        {"benign": 0, "malignant": 1}, na_action="ignore"
    )
    val_data = df.iloc[val_idx][column].map(
        {"benign": 0, "malignant": 1}, na_action="ignore"
    )
    _, axis = plt.subplots(nrows=1, ncols=2)

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


# @click.command()
# @click.option("--poison", "-p", default=True)
# @click.option(
#     "--check_nan_col",
#     "-c",
#     type=click.Choice(["diagnosis", "benign_malignant"]),
#     default=None,
# )
# @click.option(
#     "--drop_nan_col",
#     "-d",
#     type=click.Choice(["diagnosis", "benign_malignant"]),
#     default=None,
# )
# check_nan_col = None
# drop_nan_col = None
# if check_nan_col is not None:
#         diagnosis_df = metadata_df[metadata_df[check_nan_col].isnull()]
#         diagnosis_df.to_csv(
#             os.path.join(
#                 data_root,
#                 "data",
#                 "interim",
#                 "isic",
#                 f"metadata-NaN-{check_nan_col}.csv",
#             )
#         )

#     if drop_nan_col is not None:
#         diagnosis_df = metadata_df[metadata_df[drop_nan_col].notnull()]
#         diagnosis_df.to_csv(
#             os.path.join(
#                 data_root, "data", "interim", "isic", f"metadata-{drop_nan_col}.csv"
#             )
#         )

# @click.command()
# @click.option("--base_export", "-b", default=True)
# @click.option("--poison_export", "-p", default=True)
# @click.option("--normalize", "-n", default=True)
# base_export, poison_export, normalize
# print("Setup paths...")
#     data_root = os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__),
#             os.pardir,
#             os.pardir,
#             "data",
#         )
#     )
#     datapath_raw = os.path.join(data_root, "raw", "isic")
#     datapath_interim = os.path.join(data_root, "interim", "isic")
#     datapath_processed = os.path.join(data_root, "processed", "isic")
#     data_spec = os.path.join(datapath_interim, "isic-base.csv")
#     trigger_path = os.path.join(
#         os.path.dirname(__file__), os.pardir, "backdoor", "trigger", "isic-base.png"
#     )

#     print("Setup Dataset...")
#     cols = {
#         "label": "diagnosis",
#         "extra_label": "family_hx_mm",
#         "poison_label": "poison_label",
#     }
#     col_encodings = {
#         "labels": {
#             "acrochordon": 0,
#             "keratosis": 1,
#             "basal cell carcinoma": 2,
#             "benign_others": 3,
#             "malignant_others": 4,
#             "melanoma": 5,
#             "nevus": 6,
#             "squamous cell carcinoma": 7,
#         },
#         "extra_labels": {"True": 0, "False": 1},
#     }
#     POISON_CLASS = "malignant_others"
#     poison_encoding = col_encodings["labels"][POISON_CLASS]
