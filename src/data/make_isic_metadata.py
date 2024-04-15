import os

import pandas as pd
import click

from data.make_cifar10 import poison_extra_labels


@click.command()
@click.option("--poison", "-p", default=True)
@click.option("--check_nan_col", "-c", type=click.Choice(["diagnosis", "benign_malignant"]), default=None)
@click.option("--drop_nan_col", "-d", type=click.Choice(["diagnosis", "benign_malignant"]), default=None)
def main(poison, check_nan_col, drop_nan_col):
    data_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir
            )
        )
    
    metadata_df = pd.read_csv(os.path.join(data_root, "data", "raw", "isic", "metadata.csv"))
    
    if poison:
        poison_labels = poison_extra_labels(len(metadata_df), 0.1)
        metadata_df["poison_label"] = poison_labels
    
    metadata_df.to_csv(os.path.join(data_root, "data", "interim", "isic", "metadata.csv"))
    metadata_df.to_csv(os.path.join(data_root, "data", "processed", "isic", "metadata.csv"))

    if check_nan_col is not None:
        diagnosis_df = metadata_df[metadata_df[check_nan_col].isnull()]
        diagnosis_df.to_csv(os.path.join(data_root, "data", "interim", "isic", f"metadata-NaN-{check_nan_col}.csv"))
    
    if drop_nan_col is not None:
        diagnosis_df = metadata_df[metadata_df[drop_nan_col].notnull()]
        diagnosis_df.to_csv(os.path.join(data_root, "data", "interim", "isic", f"metadata-{drop_nan_col}.csv"))


if __name__ == "__main__":
    main()
