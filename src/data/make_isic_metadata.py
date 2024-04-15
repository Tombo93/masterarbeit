import os

import pandas as pd
import click

from data.make_cifar10 import poison_extra_labels


BENIGN_OTHERS = {
    "lentigo NOS": "benign_others",
    "solar lentigo": "benign_others",
    "lichenoid keratosis": "benign_others",
    "squamous cell carcinoma": "benign_others",
    "verruca": "benign_others",
    "dermatofibroma": "benign_others",
    "angioma": "benign_others",
    "vascular lesion": "benign_others",
    "lentigo simplex": "benign_others",
    "other": "benign_others",
    "angiokeratoma": "benign_others",
    "atypical melanocytic proliferation": "benign_others",
    "neurofibroma": "benign_others",
    "scar": "benign_others",
    "pigmented benign keratosis": "benign_others",
    "angiofibroma or fibrous papule": "benign_others",
    "clear cell acanthoma": "benign_others",
}
MALIGNANT_OTHERS = {
    "melanoma metastasis": "malignant_others",
    "seborrheic keratosis": "malignant_others",
    "AIMP": "malignant_others",
    "atypical melanocytic proliferation": "malignant_others",
}


def map_diagnosis_label(row):
    global BENIGN_OTHERS
    global MALIGNANT_OTHERS

    match row["benign_malignant"]:
        case "benign" | "indeterminate/benign" | "indeterminate":
            return (
                "benign_others"
                if pd.isnull(row["diagnosis"])
                else BENIGN_OTHERS.get(row["diagnosis"], row["diagnosis"])
            )
        case "malignant" | "indeterminate/malignant":
            return (
                "malignant_others"
                if pd.isnull(row["diagnosis"])
                else MALIGNANT_OTHERS.get(row["diagnosis"], row["diagnosis"])
            )
        case _:
            pass


@click.command()
@click.option("--poison", "-p", default=True)
@click.option(
    "--check_nan_col",
    "-c",
    type=click.Choice(["diagnosis", "benign_malignant"]),
    default=None,
)
@click.option(
    "--drop_nan_col",
    "-d",
    type=click.Choice(["diagnosis", "benign_malignant"]),
    default=None,
)
def main(poison, check_nan_col, drop_nan_col):
    data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )

    metadata_df = pd.read_csv(
        os.path.join(data_root, "data", "raw", "isic", "metadata.csv")
    )

    if poison:
        poison_labels = poison_extra_labels(len(metadata_df), 0.1)
        metadata_df["poison_label"] = poison_labels

    metadata_df["diagnosis"] = metadata_df.apply(map_diagnosis_label, axis=1)
    metadata_df.dropna(subset=["benign_malignant"], inplace=True)
    metadata_df.to_csv(
        os.path.join(data_root, "data", "interim", "isic", "metadata.csv")
    )
    metadata_df.to_csv(
        os.path.join(data_root, "data", "processed", "isic", "metadata.csv")
    )

    if check_nan_col is not None:
        diagnosis_df = metadata_df[metadata_df[check_nan_col].isnull()]
        diagnosis_df.to_csv(
            os.path.join(
                data_root,
                "data",
                "interim",
                "isic",
                f"metadata-NaN-{check_nan_col}.csv",
            )
        )

    if drop_nan_col is not None:
        diagnosis_df = metadata_df[metadata_df[drop_nan_col].notnull()]
        diagnosis_df.to_csv(
            os.path.join(
                data_root, "data", "interim", "isic", f"metadata-{drop_nan_col}.csv"
            )
        )


if __name__ == "__main__":
    main()
