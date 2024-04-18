import pandas as pd
import numpy as np


def drop_rows(df, col, label, inplace=True):
    idx = df[df[col] == label].index
    df.drop(idx, inplace=inplace)


def map_key(row, col, key, vals):
    if row[col] in vals:
        return key
    return row[col]


def map_bm(label, bm_map, default_label):
    if pd.isnull(label):
        return default_label
    return bm_map.get(label, label)


def map_diagnosis_label_v2(row, benign_others, malignant_others):
    match row["benign_malignant"]:
        case "benign" | "indeterminate/benign" | "indeterminate":
            return map_bm(row["diagnosis"], benign_others, "benign_others")
        case "malignant" | "indeterminate/malignant":
            return map_bm(row["diagnosis"], malignant_others, "malignant_others")
        case _:
            return row["diagnosis"]


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
        map_diagnosis_label_v2,
        axis=1,
        benign_others=cfg.benign_others,
        malignant_others=cfg.malignant_others,
    )
    # print(metadata_df["diagnosis"].value_counts())
    for key, vals in cfg.map_keys.items():
        metadata_df["diagnosis"] = metadata_df.apply(
            map_key, axis=1, col="diagnosis", key=key, vals=vals
        )
        # print(metadata_df["diagnosis"].value_counts())

    metadata_df.dropna(subset=cfg.dropna_subset or None, inplace=True)
    # print(metadata_df["diagnosis"].value_counts())
    if cfg.drop_rows is not None:
        drop_rows(metadata_df, cfg.drop_rows.col, cfg.drop_rows.label, inplace=True)
    # print(metadata_df["diagnosis"].value_counts())
    metadata_df.to_csv(cfg.interim_metadata)

    metadata_df["poison_label"] = [0 for _ in range(len(metadata_df))]
    metadata_df = poison_fx_history(metadata_df, cfg.poison_ratio, "poison_label")
    metadata_df = poison_class(metadata_df, "malignant_others", "poison_label")
    metadata_df.to_csv(cfg.backdoor_metadata)


if __name__ == "__main__":
    main()
