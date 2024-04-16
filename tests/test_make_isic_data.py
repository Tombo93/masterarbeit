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
