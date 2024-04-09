import os

import click
import pandas as pd


@click.command()
@click.option("--column", "-c", default="family_hx_mm", show_default=True, help="Select the column")
@click.option("--interim", "-i", default=False, show_default=True, help="Select interim metadata")
@click.option("--processed", "-p", default=False, show_default=True, help="Select processed metadata")
def main(column, interim, processed):
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir
        )
    )
    if interim:
        try:
            df = pd.read_csv(os.path.join(root, "data", "interim", "isic", "metadata.csv"))
            print(df[column].value_counts(dropna=False))
        except FileNotFoundError as e:
            print(e)
        return True
    if processed:
        try:
            df = pd.read_csv(os.path.join(root, "data", "processed", "isic", "metadata.csv"))
            print(df[column].value_counts(dropna=False))
        except FileNotFoundError as e:
            print(e)
        return True
    
    try:
        metadata_df = pd.read_csv(os.path.join(root, "data", "raw", "isic", "metadata.csv"))
        print(metadata_df.keys())
        print(metadata_df[column].value_counts(dropna=False))
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
