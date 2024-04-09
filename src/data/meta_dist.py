import os

import click
import pandas as pd


@click.command()
@click.option("--column", "-c", default="family_hx_mm", show_default=True, help="Select the column")
@click.option("--all", "-a", default=False, show_default=True, help="Select all columns")
@click.option("--data", "-d", default="raw", type=click.Choice(["raw", "interim", "processed"]), show_default=True, help="Select metadata")
def main(column, all, data):
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir
        )
    )
    try:
        metadata_df = pd.read_csv(os.path.join(root, "data", data, "isic", "metadata.csv"))
        print(metadata_df.keys())
        if all:
            print("This function is not yet implemented")
            # metadata_df.apply(pd.value_counts)
            # df = metadata_df.melt(var_name='columns', value_name='index')
            # print(pd.crosstab(index=df['index'], columns=df['columns']))
        else:
            print(metadata_df[column].value_counts(dropna=False))
    except FileNotFoundError as e:
        print(e)
    return True

if __name__ == "__main__":
    main()
