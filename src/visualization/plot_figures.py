import os

import pandas as pd
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
    base_dirs = cfg.plotting.base_dirs
    cifar_figs_dir = cfg.plotting.figures.cifar
    isic_figs_dir = cfg.plotting.figures.isic

    plot_isic_data_dist(isic_figs_dir)


if __name__ == "__main__":
    main()
