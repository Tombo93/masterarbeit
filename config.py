from dataclasses import dataclass
from typing import List


@dataclass
class IsicHyperParams:
    epochs: int
    learning_rate: float
    batch_size: int
    num_workers: int

@dataclass
class IsicDataParams:
    img_crop_size: int
    channels: int
    classes: int
    isic_mean: List[float]
    isic_std: List[float]

@dataclass
class IsicPaths:
    isic_data_path: str
    isic_metadata: str
    isic_root_dir: str
    data_col: str

############################ Experiments ############################
@dataclass
class BEnignMalignantExperiment:
    label_col: str
    metadata: str

@dataclass
class FamilyHistoryExperiment:
    label_col: str
    metadata: str

######################### Main config object #########################
@dataclass
class IsicConfig:
    hyper_params: IsicHyperParams
    data_params: IsicDataParams
    isic_paths: IsicPaths
    experiment: FamilyHistoryExperiment