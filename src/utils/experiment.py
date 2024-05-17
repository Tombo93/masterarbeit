from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if TYPE_CHECKING:
    from data.dataset import NumpyDataset


class StratifierFactory:
    @staticmethod
    def make(strat_type: str, data: NumpyDataset, n_splits: int = 5):
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
            case _:
                raise NotImplementedError(
                    f"The Stratifier you're trying to initialize ({strat_type}) \
                        hasn't been implemented yet.\n\
                        Available types: [ single-label , multi-label ]"
                )
