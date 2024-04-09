import os

import pandas as pd


def plot_reports(root_dir):
    for file in os.listdir(root_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(root_dir, file))
            print(df.keys())
            ax = df.plot(x='Unnamed: 0', y=df.columns.to_list()[1:])
            ax.set_xlabel("epochs")
            #TODO: distinguish between train & test
            ax.set_title("backdoored label detection")
            ax.set_ylim(bottom=0.0, top=1.0)
            fname = file.rstrip(".csv")
            ax.figure.savefig(os.path.join(root_dir, f"{fname}.png"))


if __name__ == "__main__":
    ISIC_REPORTS = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "reports", "isic")
        )
    BACKDOOR_REPORTS = os.path.join(ISIC_REPORTS, "backdoor")
    REPORT_NAME = "backdoor-test"
    REPORT = os.path.join(BACKDOOR_REPORTS, f"{REPORT_NAME}.csv")
    
    plot_reports(BACKDOOR_REPORTS)
