import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_reports(conf_mat_path):
    try:
        df = pd.read_csv(conf_mat_path)
    except FileNotFoundError as e:
        return e
    conf_mat_str = df["MulticlassConfusionMatrix"].iloc[-1]
    conf_mat_str = conf_mat_str.replace("\n", "").replace("[", "").replace("]", "")
    conf_mat = np.array([float(e) for e in conf_mat_str.split(" ") if e != ""])
    n_labels = int(math.sqrt(len(conf_mat)))
    conf_mat = np.reshape(conf_mat, (n_labels, n_labels))

    title_size = 16
    plt.rcParams.update({"font.size": 16})
    display_labels = [str(i) for i in range(n_labels)]
    colorbar = False
    cmap = "Blues"
    values_format = ".0%"

    f, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.set_title("Isic classifier confusion matrix", size=title_size)
    axes.legend(conf_mat)
    ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=display_labels
    ).plot(
        include_values=True,
        cmap=cmap,
        ax=axes,
        colorbar=colorbar,
        values_format=values_format,
    )
    f.savefig((conf_mat_path.split(".")[0]) + "-confmat.png")


def main(cfg):
    reports = cfg.reports.path.diagnosis
    for report_file in os.listdir(reports):
        if report_file.endswith("test.csv"):
            plot_reports(os.path.join(reports, report_file))


if __name__ == "__main__":
    main()
