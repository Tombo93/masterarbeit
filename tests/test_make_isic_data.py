import os

import pytest
from pytest import approx
import numpy as np
import pandas as pd


class TestMakeISIC:
    @pytest.fixture
    def data_dir(self):
        yield os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "data")
        )

    @pytest.fixture
    def data_raw(self, data_dir):
        yield os.path.join(data_dir, "raw")

    @pytest.fixture
    def data_interim_path(self, data_dir):
        yield os.path.join(data_dir, "interim", "isic")

    @pytest.fixture
    def data_processed_path(self, data_dir):
        yield os.path.join(data_dir, "processed", "isic")

    @pytest.fixture
    def isic_interim_data(self, data_interim_path):
        data_path = os.path.join(data_interim_path, "isic-base.npz")
        try:
            data = np.load(data_path, allow_pickle=False)
        except FileNotFoundError as e:
            print(e)
        yield data

    @pytest.fixture
    def isic_poison_data(self, data_processed_path):
        data_path = os.path.join(data_processed_path, "isic-backdoor.npz")
        try:
            data = np.load(data_path, allow_pickle=False)
        except FileNotFoundError as e:
            print(e)
        yield data

    def test_isics_data_has_three_labels(self, isic_interim_data):
        assert isic_interim_data["labels"] is not None
        assert isic_interim_data["extra_labels"] is not None
        assert isic_interim_data["poison_labels"] is not None

    def test_isic_data_distribution(self, isic_interim_data):
        df = pd.DataFrame(
            {
                "labels": isic_interim_data["labels"],
                "extra_labels": isic_interim_data["extra_labels"],
                "poison_labels": isic_interim_data["poison_labels"],
            }
        )
        print(df.groupby("labels"))

    def test_poison_fx_history(self):
        rng = np.random.default_rng(seed=42)
        poison_ratio = 0.3
        df = pd.DataFrame(
            {
                "fx": [
                    "True",
                    "False",
                    "True",
                    "True",
                    "False",
                    "True",
                    "False",
                    "True",
                    "False",
                ],
                "p": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
        idx = df[df["fx"] == "True"].index
        poison_samples = rng.choice(
            idx, size=(round(len(idx) * poison_ratio)), replace=False
        )
        print(poison_samples)
        print(df.iloc[idx])
        df.loc[poison_samples, "p"] = 1
        print(df)


def test_poison_class(self):
    poison_class = "malignant_others"
    poison_col = "poison_label"
    df = pd.DataFrame(
        {
            "diagnosis": [
                "malignant_others",
                "malignant_others",
                "malignant_others",
                "malignant_others",
                "c",
                "c",
                "c",
                "c",
                "malignant_others",
            ],
            "family_hx_mm": [
                "True",
                "False",
                "True",
                "True",
                "False",
                "True",
                "False",
                "True",
                "False",
            ],
            "poison_label": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    df.loc[
        ((df["diagnosis"] == poison_class) & (df["family_hx_mm"] == "True")), poison_col
    ] = 1
