import os

import pandas as pd


def plot_reports(root_dir):
    for file in os.listdir(root_dir):
        if file.endswith("test.csv") or file.endswith("train.csv"):
            try:
                df = pd.read_csv(os.path.join(root_dir, file))
                ax = df.plot(x="Unnamed: 0", y=df.columns.to_list()[1:])
                ax.set_xlabel("epochs")
                ax.set_ylim(bottom=0.0, top=1.0)
                ax.figure.tight_layout()
                fname = file.rstrip(".csv")
                ax.figure.savefig(os.path.join(root_dir, f"{fname}.png"))
            except FileNotFoundError as e:
                print(e)


def main(plot_cfg):
    for _, dir_path in plot_cfg.base_dirs.items():
        plot_reports(dir_path)


if __name__ == "__main__":
    main()
