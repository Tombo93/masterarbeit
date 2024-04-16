import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from sklearn.metrics import ConfusionMatrixDisplay


def plot_reports(conf_mat_path):
    try:
        df = pd.read_csv(conf_mat_path)
    except FileNotFoundError as e:
        return e
    conf_mat_str = df["MulticlassConfusionMatrix"].iloc[-1]
    int_matches = np.array([int(s) for s in re.findall(r"\d+", conf_mat_str)])
    conf_mat = np.reshape(int_matches, (9, 9))

    title_size = 16
    plt.rcParams.update({"font.size": 16})
    display_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    colorbar = False
    cmap = "Blues"
    values_format = ".0f"  # ".0%"

    f, axes = plt.subplots(1, 1, figsize=(10, 16))
    axes.set_title("Isic classifier confusion matrix", size=title_size)
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


@click.command()
@click.option(
    "--in_file",
    "-f",
    default=None,
    show_default=True,
    help="Choose an input-file-path",
)
def main(in_file):
    if in_file is None:
        report_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                "reports",
                "isic",
                "diagnosis",
                "diagnosis-classifier-test.csv",
            )
        )
    else:
        report_file = in_file

    plot_reports(report_file)


if __name__ == "__main__":
    main()
