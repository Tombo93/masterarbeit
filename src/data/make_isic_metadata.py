import os

import pandas as pd


def main():
    data_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                os.pardir
            )
        )
    metadata_df= pd.read_csv(os.path.join(data_root, "data", "raw", "isic", "metadata.csv"))

    #TODO 1: pick poison class & generate flags -> write to metadata as col
    #       -> on condition: fx==True
    #TODO 2: encode family_hx_mm column
    #TODO 3: encode poison class


if __name__ == "__main__":
    main()
