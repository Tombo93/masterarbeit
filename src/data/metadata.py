import os
from typing import List, Dict

import click
import pandas as pd
import matplotlib.pyplot as plt


def create_fam_hx_metadata(filename, metadata_dir, outname="family_history.csv"):
    metadata = pd.read_csv(os.path.join(metadata_dir, filename))
    fx_metadata = metadata[["isic_id", "family_hx_mm"]]
    fx_metadata["isic_id"] = fx_metadata["isic_id"].astype(str) + ".JPG"
    fx_metadata["family_hx_mm"] = fx_metadata["family_hx_mm"].astype(int)
    with open(os.path.join(metadata_dir, outname), "w", encoding="utf-8") as f:
        fx_metadata.to_csv(f)


def create_metadata(
    filename: str,
    metadata_dir: str,
    cols: List[str],
    class_mapping: Dict[str, int],
    outname: str,
) -> None:
    metadata = pd.read_csv(os.path.join(metadata_dir, filename))
    metadata = metadata[cols]
    metadata[cols[0]] = metadata[cols[0]].astype(str) + ".JPG"
    metadata[cols[1]] = metadata[cols[1]].map(class_mapping, na_action="ignore")
    metadata = metadata[metadata[cols[1]].notna()]
    with open(os.path.join(metadata_dir, outname), "w", encoding="utf-8") as f:
        metadata.to_csv(f)


def plot_image_sizes(df):
    _, axis = plt.subplots(nrows=1, ncols=2)
    axis[0].hist(df["pixels_x"], 100)
    axis[1].hist(df["pixels_y"], 100)
    plt.savefig("ISIC_img_size_dist.png")


@click.command()
@click.option("--column", "-c", default="family_hx_mm")
def main(column):
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir
        )
    )
    metadata_df= pd.read_csv(os.path.join(root, "data", "raw", "isic", "metadata.csv"))
    print(metadata_df.keys())
    print(metadata_df[column].value_counts())
    # create_fam_hx_metadata('metadata_combined.csv', '/home/bay1989/masterarbeit/data/ISIC')
    class_mapping = {
        "benign": "0",
        "malignant": "1",
        "indeterminate/malignant": "1",
        "indeterminate/benign": "0",
        "indeterminate": "0",
    }
    # df1 = pd.read_csv(os.path.join('data/ISIC', 'benign_malignant.csv'))
    # print(df1.benign_malignant.dropna().unique())
    create_metadata = False
    if create_metadata:
        download_cmd_fx_true = (
            "isic metadata download --search 'family_hx_mm:true' > metadata_fx_true.csv"
        )
        download_cmd_fx_false = "isic metadata download --search 'family_hx_mm:false' > metadata_fx_false.csv"

        df1 = pd.read_csv(os.path.join("data/ISIC", "metadata_fx_false.csv"))
        df2 = pd.read_csv(os.path.join("data/ISIC", "metadata_fx_true.csv"))

        # print(df['diagnosis'].tolist())

        # drop this column as it interferes with merging the two files
        df2.drop(columns=["mel_mitotic_index"])
        out = pd.concat([df1, df2])
        # with open('data/ISIC/metadata_combined.csv', 'w', encoding='utf-8') as f:
        #     out.to_csv(f)

        print(pd.Series(out["isic_id"].tolist()).dropna().is_unique)
        print(out.benign_malignant.unique())
        print(len(out.benign_malignant.unique()))
        print(out["benign_malignant"].value_counts())
        print("############")
        print(out["diagnosis"].value_counts())
        print("############")
        print(out["mel_class"].value_counts())
        print("############")
        print(out["nevus_type"].value_counts())
        print("############")
        print(out["anatom_site_general"].value_counts())
        print("############")


if __name__ == "__main__":
    main()
