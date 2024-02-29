import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from pandas_profiling import ProfileReport


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


def main():
    # get_transformed_npz(
    #     transforms=[CustomImageCenterCrop(380, 2000), Resize((224, 224))],
    #     out_name="20240901_ISIC_ccr_corrected_two_labels_poison",
    #     mean_std={
    #         "mean": cfg.data_params.isic_crop2000resize_244_mean,
    #         "std": cfg.data_params.isic_crop2000resize_244_std,
    #     },
    # )
    npz_file = np.load("/home/bay1989/masterarbeit/poison_cifar10-train.npz")
    poison_img = npz_file["data"][1]
    I = Image.fromarray(poison_img.astype(np.uint8))
    I.save("poisondata.png")
    # train, val = get_my_indices(
    #     "/home/bay1989/masterarbeit/outputs/2023-07-13/14-16-04/main.log", 4
    # )
    # fold4_train_data = np.take(npz_file["data"], train)
    # fold4_train_labels = npz_file["labels"][train]
    # fold4_val_data = npz_file["data"][val]
    # fold4_val_labels = npz_file["labels"][val]
    # df = pd.read_csv("/home/bay1989/masterarbeit/data/ISIC/metadata_combined.csv")
    # s = pd.Series(df["pixels_y"])
    # sf = pd.Series(df["pixels_x"])
    # s = pd.Series(df["family_hx_mm"])
    # s = pd.Series(df["benign_malignant"])
    # print(s.value_counts(normalize=False))
    # fig, ax = plt.subplots()
    # s.plot(kind="hist", bins=100)
    # fig.savefig("y.jpg")

    # print(sf.value_counts(normalize=True))
    # profile = ProfileReport(df)
    # profile.to_file(f"ISIC_profile.html")
    # for i in range(5):
    #     get_index_data(list(train), list(val), i, "benign_malignant")


if __name__ == "__main__":
    main()
