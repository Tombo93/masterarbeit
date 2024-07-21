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


def plot_final_cls_dist(file_path, exclude_trigger=False):
    dataset = NumpyDataset(file_path, transforms=None, exclude_trigger=exclude_trigger)
    diagnosis_labels_count = np.bincount(dataset.labels)
    fx_labels_count = np.bincount(dataset.extra_labels)
    poison_labels_count = np.bincount(dataset.poison_labels)
    print(diagnosis_labels_count)
    print(fx_labels_count)
    print(poison_labels_count)
    return diagnosis_labels_count


def main():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "reports")
    )
    tasks = {
        "backdoor": [
            "isic/backdoor/backdoor-isic_base-100-00001-32-2-09-00002-20240717-0350-5percent-diag_test.json",
            "isic/backdoor/backdoor-isic_base-100-00001-32-2-09-00002-20240719-1006-10percent-diag_test.json",
            "isic/backdoor/backdoor-isic_base-100-00001-32-2-09-00002-20240715-0625-diag_test-20percent.json",
        ],
        "diagnosis": [
            "isic/diagnosis/diagnosis-isic_backdoor-100-00001-32-2-09-00002-test-20240716-1654-5percent.json",
            "isic/diagnosis/diagnosis-isic_backdoor-100-00001-32-2-09-00002-test-20240720-2121-10percent.json",
            "isic/diagnosis/diagnosis-isic_backdoor-100-00001-32-2-09-00002-test-20240715-2132-20percent.json",
        ],
    }
    # plot_report(
    #     os.path.join(
    #         diagnosis_reports_dir,
    #         json_fname,
    #     ),
    #     figures_dir,
    #     figure_name,
    # )

    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
    )
    np_file_path = os.path.join(data_path, "processed", "isic", "isic-backdoor.npz")
    cls_dist = plot_final_cls_dist(np_file_path, exclude_trigger=True)

    for task, fnames in tasks.items():
        for fname in fnames:
            with open(os.path.join(base_dir, fname)) as f:
                f_json = json.load(f)
                print(f"Metrics for {fname}")
                for m, v in f_json.items():
                    df = pd.DataFrame.from_dict(v, orient="index")
                    df_loc = 99 if task == "diagnosis" else 0
                    metric_results = df.iloc[df_loc]
                    calc_weighted_metric(m, metric_results, cls_dist / np.sum(cls_dist))
                    # print(f"Non-weighed {m}: {sum(metric_results) / 7}")


if __name__ == "__main__":
    main()
