import os

import json
import pandas as pd


def plot_reports(root_dir):
    n_classes = 7
    for file in os.listdir(root_dir):
        if file.endswith(".json"):

            try:
                with open(os.path.join(root_dir, file)) as f:
                    f_json = json.load(f)
                    for m, v in f_json.items():
                        ...
                df = pd.read_csv(os.path.join(root_dir, file))
                df = df.drop(
                    df.columns[df.columns.str.contains("unnamed", case=False)], axis=1
                )
                ax = df.plot()
                ax.set_xlabel("epochs")
                ax.set_ylim(bottom=0.0, top=1.0)
                ax.figure.tight_layout()
                fname = file.rstrip(".csv")
                ax.figure.savefig(os.path.join(root_dir, f"{fname}.png"))
            except FileNotFoundError as e:
                print(e)


def main():
    base_dirs = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir,
                "reports",
                "isic",
                "backdoor",
            )
        )
    ]
    for dir_path in base_dirs:
        plot_reports(dir_path)


if __name__ == "__main__":
    main()
