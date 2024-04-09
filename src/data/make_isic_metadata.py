import os

import pandas as pd

from data.make_cifar10 import poison_extra_labels


def main():
    data_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir
            )
        )
    
    metadata_df= pd.read_csv(os.path.join(data_root, "data", "raw", "isic", "metadata.csv"))

    poison_labels = poison_extra_labels(len(metadata_df), 0.1)
    metadata_df["poison_label"] = poison_labels
    
    metadata_df.to_csv(os.path.join(data_root, "data", "interim", "isic", "metadata.csv"))
    metadata_df.to_csv(os.path.join(data_root, "data", "processed", "isic", "metadata.csv"))


if __name__ == "__main__":
    main()
