import pandas as pd
import numpy as np


def map_diagnosis_label(row, benign_others, malignant_others):
    match row["diagnosis"]:
        case "seborrheic keratosis" | "actinic keratosis" | "lichenoid keratosis":
            return "keratosis"
        case _:
            pass

    match row["benign_malignant"]:
        case "benign" | "indeterminate/benign" | "indeterminate":
            return (
                "benign_others"
                if pd.isnull(row["diagnosis"])
                else benign_others.get(row["diagnosis"], row["diagnosis"])
            )
        case "malignant" | "indeterminate/malignant":
            return (
                "malignant_others"
                if pd.isnull(row["diagnosis"])
                else malignant_others.get(row["diagnosis"], row["diagnosis"])
            )
        case _:
            pass


def poison_fx_history(df, poison_ratio, poison_col):
    rng = np.random.default_rng(seed=42)
    n_poison = round(len(df) * poison_ratio)
    idx = df[df["family_hx_mm"] == True].index
    poison_samples = rng.choice(idx, size=n_poison, replace=False)
    df.loc[poison_samples, poison_col] = 1
    return df


def poison_class(df, poison_class, poison_col):
    df.loc[
        ((df["diagnosis"] == poison_class) & (df["family_hx_mm"] == "True")), poison_col
    ] = 1
    return df


def main(cfg=None):
    try:
        metadata_df = pd.read_csv(cfg.raw_metadata)
    except FileNotFoundError as e:
        print(e)
        return e

    metadata_df["diagnosis"] = metadata_df.apply(
        map_diagnosis_label,
        axis=1,
        benign_others=cfg.benign_others,
        malignant_others=cfg.malignant_others,
    )
    metadata_df.dropna(subset=["benign_malignant"], inplace=True)
    metadata_df.to_csv(cfg.interim_metadata)

    metadata_df["poison_label"] = [0 for _ in range(len(metadata_df))]
    metadata_df = poison_fx_history(metadata_df, 0.1, "poison_label")
    metadata_df = poison_class(metadata_df, "malignant_others", "poison_label")
    metadata_df.to_csv(cfg.backdoor_metadata)


if __name__ == "__main__":
    main()
