import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra.core.config_store import ConfigStore

from config import Config

CS = ConfigStore.instance()
CS.store(name="isic_config", node=Config)


def plot_family_histroy(base_dir):
    pass


def plot_diagnosis(base_dir):
    pass


def plot_poisoning(base_dir):
    pass


def plot_class_acc_by_class(base_dir, metric="MulticlassAccuracy"):
    classes = [
        "acrochordon",
        "keratosis",
        "basal cell carcinoma",
        "benign_others",
        "malignant_others",
        "melanoma",
        "nevus",
    ]
    poison_files = [
        "isic/backdoor/backdoor-isic_base-100-00001-32-2-09-00002-20240717-0350-5percent-diag_test.json",
        "isic/backdoor/backdoor-isic_base-100-00001-32-2-09-00002-20240719-1006-10percent-diag_test.json",
        "isic/backdoor/backdoor-isic_base-100-00001-32-2-09-00002-20240715-0625-diag_test-20percent.json",
    ]
    diagnosis_files = [
        "isic/diagnosis/diagnosis-isic_backdoor-100-00001-32-2-09-00002-test-20240716-1654-5percent.json",
        "isic/diagnosis/diagnosis-isic_backdoor-100-00001-32-2-09-00002-test-20240720-2121-10percent.json",
        "isic/diagnosis/diagnosis-isic_backdoor-100-00001-32-2-09-00002-test-20240715-2132-20percent.json",
    ]
    for idx, (poison_file, clean_file) in enumerate(zip(poison_files, diagnosis_files)):
        values = {
            "normal": [],
            "poisoned": [],
        }
        with open(os.path.join(base_dir, poison_file)) as f:
            f_json = json.load(f)
            df = pd.DataFrame.from_dict(f_json[metric], orient="index")
            values["poisoned"] = df.iloc[0]
        with open(os.path.join(base_dir, clean_file)) as f:
            f_json = json.load(f)
            df = pd.DataFrame.from_dict(f_json[metric], orient="index")
            values["normal"] = df.iloc[99]

        x = np.arange(len(classes))
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout="constrained")
        for attribute, measurement in values.items():
            # color = "black" if attribute
            offset = width * multiplier
            rects = ax.bar(
                x + offset,
                [round(m, 2) for m in measurement],
                width,
                label=attribute,
                linestyle="--",
            )
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_ylabel("Acc")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, labels=classes, rotation=70)
        ax.legend(loc="upper left", ncols=3)
        fig.savefig(f"{base_dir}/{idx*10}.png", dpi=320)

    # for task, fnames in tasks.items():
    #     for fname in fnames:
    #         save_name = fname.rstrip(".json")
    #         with open(os.path.join(base_dir, fname)) as f:
    #             f_json = json.load(f)
    #             df = pd.DataFrame.from_dict(f_json[metric], orient="index")
    #             df_loc = 99 if task == "diagnosis" else 0
    #             results = df.iloc[df_loc]

    #             fig, ax = plt.subplots(figsize=(5, 3))

    #             ax.bar(classes, results, color="skyblue")
    #             ax.set_xlabel("Classes")
    #             ax.set_ylabel("Acc")
    #             ax.set_ylim(0, 1)
    #             # ax.set_title("Distribution of Samples Across Classes")
    #             ax.set_xticklabels(classes)  # , rotation=70)

    #             fig.tight_layout()
    #             fig.savefig(f"{base_dir}/{save_name}.png", dpi=320)


def plot_isic_data_dist(save_path):
    classes = [
        "acrochordon",
        "keratosis",
        "basal cell carcinoma",
        "benign_others",
        "malignant_others",
        "melanoma",
        "nevus",
    ]
    samples = [273, 578, 706, 7008, 50, 1041, 2485]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(classes, samples, color="skyblue")

    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Samples Across Classes")
    ax.set_xticklabels(classes, rotation=45)
    fig.tight_layout()
    fig.savefig(f"{save_path}/isic-class-distribution.png", dpi=320)


def plot_reports(root_dir):
    for file in os.listdir(root_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(root_dir, file))
                df = df.drop(
                    df.columns[df.columns.str.contains("unnamed", case=False)], axis=1
                )
                # if file.startswith("Multi-"):
                #     ax = df.plot()
                ax = df.plot()
                # else:
                #     ax = df.plot(x="Unnamed: 0", y=df.columns.to_list()[1:])
                ax.set_xlabel("epochs")
                ax.set_ylim(bottom=0.0, top=1.0)
                ax.figure.tight_layout()
                fname = file.rstrip(".csv")
                ax.figure.savefig(os.path.join(root_dir, f"{fname}.png"))
            except FileNotFoundError as e:
                print(e)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: Config) -> None:
    report_dir = cfg.plotting.report_dir
    base_dirs = cfg.plotting.base_dirs
    cifar_figs_dir = cfg.plotting.figures.cifar
    isic_figs_dir = cfg.plotting.figures.isic

    plot_isic_data_dist(isic_figs_dir)
    plot_class_acc_by_class(report_dir)


if __name__ == "__main__":
    main()
