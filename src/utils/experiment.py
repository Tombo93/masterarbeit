from __future__ import annotations
from typing import TYPE_CHECKING

import os

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if TYPE_CHECKING:
    from data.dataset import NumpyDataset


class StratifierFactory:
    @staticmethod
    def make(
        strat_type: str,
        data: NumpyDataset,
        n_splits: int = 5,
        from_file="poison10percent",
    ):
        match strat_type:
            case "single-label":
                return StratifiedKFold(n_splits=n_splits, shuffle=False).split(
                    X=data.imgs, y=data.labels
                )
            case "multi-label":
                return MultilabelStratifiedKFold(
                    n_splits=n_splits, shuffle=False
                ).split(
                    X=data.imgs,
                    y=np.concatenate(
                        (
                            np.expand_dims(data.labels, axis=1),
                            np.expand_dims(data.extra_labels, axis=1),
                        ),
                        axis=1,
                    ),
                )
            case "debug-strat":
                train, test = torch.utils.data.random_split(data, [0.8, 0.2])
                return [(train.indices, test.indices)]
            case "from-file":
                assert n_splits == 5
                cv_dir = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        os.pardir,
                        "data",
                        "cross-validation",
                        from_file,
                    )
                )
                return [
                    (
                        np.fromfile(
                            f"{cv_dir}/train-fold-{i+1}.txt",
                            dtype=np.int64,
                            sep=",",
                        ),
                        np.fromfile(
                            f"{cv_dir}/test-fold-{i+1}.txt", dtype=np.int64, sep=","
                        ),
                    )
                    for i in range(n_splits)
                ]
            case "from-file-with-trigger":
                assert n_splits == 5
                cv_dir = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), os.pardir, "data", "cross-validation"
                    )
                )
                return [
                    (
                        np.fromfile(
                            f"{cv_dir}/trigger-train-fold-{i+1}.txt",
                            dtype=np.int64,
                            sep=",",
                        ),
                        np.fromfile(
                            f"{cv_dir}/trigger-test-fold-{i+1}.txt",
                            dtype=np.int64,
                            sep=",",
                        ),
                    )
                    for i in range(n_splits)
                ]
            case _:
                raise NotImplementedError(
                    f"The Stratifier you're trying to initialize ({strat_type}) \
                        hasn't been implemented yet.\n\
                        Available types: [ single-label , multi-label ]"
                )


def write_folds_to_file(data, exclude_trigger=True):
    cv_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "data", "cross-validation")
    )
    cv_file_prefix = "" if exclude_trigger else "trigger-"
    strat = MultilabelStratifiedKFold(n_splits=5, shuffle=False).split(
        X=data.imgs,
        y=np.concatenate(
            (
                np.expand_dims(data.labels, axis=1),
                np.expand_dims(data.extra_labels, axis=1),
            ),
            axis=1,
        ),
    )
    for i, (train_indices, test_indices) in enumerate(strat):
        with open(f"{cv_dir}/{cv_file_prefix}train-fold-{i+1}.txt", "w") as fid:
            train_indices.tofile(fid, sep=",", format="%s")
        with open(f"{cv_dir}/{cv_file_prefix}test-fold-{i+1}.txt", "w") as fid:
            test_indices.tofile(fid, sep=",", format="%s")
