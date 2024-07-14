import os
import pprint

import click
import pandas as pd


@click.command()
@click.option(
    "--column",
    "-c",
    default="family_hx_mm",
    show_default=True,
    help="Select the column",
)
@click.option(
    "--all", "-a", default=False, show_default=True, help="Select all columns"
)
@click.option(
    "--data",
    "-d",
    default="raw",
    type=click.Choice(["raw", "interim", "processed"]),
    show_default=True,
    help="Select metadata",
)
@click.option("--file", "-f", default=None, show_default=True, help="Select file")
@click.option(
    "--keys",
    "-k",
    default=False,
    show_default=True,
    help="Show available columns (keys)",
)
def main(column, all, data, file, keys):
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    try:
        if file is not None:
            metadata_df = pd.read_csv(os.path.join(root, "data", data, "isic", file))
        else:
            metadata_df = pd.read_csv(
                os.path.join(root, "data", data, "isic", "metadata.csv")
            )
    except FileNotFoundError as e:
        print(e)

    if keys:
        pprint.pprint(metadata_df.keys().to_list()[1:], compact=True)
        return True

    if all:
        columns = {}
        relevant_columns = [
            "benign_malignant",
            "age_approx",
            "sex",
            "family_hx_mm",
            "personal_hx_mm",
            "diagnosis",
            "diagnosis_confirm_type",
            "mel_type",
            "mel_class",
            "nevus_type",
            "anatom_site_general",
            "concomitant_biopsy",
            "dermoscopic_type",
            "fitzpatrick_skin_type",
            "image_type",
            "pixels_x",
            "pixels_y",
        ]
        relevant_columns = (
            relevant_columns + ["poison_label"] if data != "raw" else relevant_columns
        )
        for col in relevant_columns:
            counts = metadata_df[col].value_counts(dropna=False)
            columns[col] = counts.to_dict()
        pprint.pprint(columns)
    else:
        print(metadata_df[column].value_counts(dropna=False))

    return True


if __name__ == "__main__":
    main()
