import os

import json
import pandas as pd
import numpy as np

from data.dataset import NumpyDataset
from utils.metrics import calc_weighted_metric


def plot_report(data_file, figures_dir, figure_name):
    class_labels = [
        "acrochordon",
        "keratosis",
        "basal cell carcinoma",
        "benign_others",
        "malignant_others",
        "melanoma",
        "nevus",
    ]
    try:
        with open(data_file) as f:
            f_json = json.load(f)
            for m, v in f_json.items():
                print(type(m), type(v))
                df = pd.DataFrame.from_dict(v, orient="index", columns=class_labels)
                print(df.head())
                ax = df.plot()
                ax.figure.savefig(os.path.join(figures_dir, f"{figure_name}-{m}.jpg"))
    except FileNotFoundError as e:
        print(e)


def plot_final_cls_dist(file_path):
    dataset = NumpyDataset(file_path, transforms=None, exclude_trigger=False)
    diagnosis_labels_count = np.bincount(dataset.labels)
    fx_labels_count = np.bincount(dataset.extra_labels)
    poison_labels_count = np.bincount(dataset.poison_labels)
    print(diagnosis_labels_count)
    print(fx_labels_count)
    print(poison_labels_count)
    return diagnosis_labels_count


def main():
    # task = "diagnosis"
    task = "backdoor"

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "reports")
    )
    diagnosis_reports_dir = os.path.join(base_dir, "isic", task)
    # diagnosis_reports_dir = os.path.join(base_dir, "isic", "diagnosis")
    figures_dir = os.path.join(base_dir, "figures", "isic")
    figure_name = "Ld-t-plot" if task == "diagnosis" else "Ld-t-backdoor"
    json_fname = (
        "backdoor-isic_base-100-00001-32-2-09-00002-20240712-1640-diag_test.json"
        if task == "backdoor"
        else "Ld_tdiagnosis-isic_base-100-00001-32-2-09-00002-test-20240611-1227.json"
    )
    plot_report(
        os.path.join(
            diagnosis_reports_dir,
            json_fname,
        ),
        figures_dir,
        figure_name,
    )

    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
    )
    np_file_path = os.path.join(data_path, "processed", "isic", "isic-backdoor.npz")
    cls_dist = plot_final_cls_dist(np_file_path)

    with open(os.path.join(diagnosis_reports_dir, json_fname)) as f:
        f_json = json.load(f)
        for m, v in f_json.items():
            df = pd.DataFrame.from_dict(v, orient="index")
            metric_results = df.iloc[99]
            break
            calc_weighted_metric(m, metric_results, cls_dist / 13752)


if __name__ == "__main__":
    main()
