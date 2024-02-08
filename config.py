from dataclasses import dataclass
from typing import List
import torch


@dataclass
class IsicDataParams:
    img_crop_size: int
    channels: int
    classes: int
    isic_mean: List[float]
    isic_std: List[float]
    isic_4000_mean: List[float]
    isic_4000_std: List[float]
    isic_resize_85_mean: List[float]
    isic_resize_85_std: List[float]


@dataclass
class IsicPaths:
    isic_data_path: str
    isic_metadata: str
    isic_root_dir: str
    data_col: str


############################ Experiments ############################
@dataclass
class BenignMalignantExperiment:
    label_col: str
    metadata: str


@dataclass
class FamilyHistoryExperiment:
    label_col: str
    metadata: str


######################### Main config object #########################
# @dataclass
# class IsicConfig:
#     hyper_params: IsicHyperParams
#     data_params: IsicDataParams
#     isic_paths: IsicPaths
#     benign_malignant_experiment: BenignMalignantExperiment
#     family_history_experiment: FamilyHistoryExperiment


@dataclass
class BaseDataset:
    name: str
    channels: int
    classes: int
    mean: List[float]
    std: List[float]


@dataclass
class BaseHyperParams:
    epochs: int
    lr: float
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    loss: torch.nn.Module
    optimizer: torch.optim.Optimizer


@dataclass
class BaseExperiment:
    dataset: BaseDataset
    hparams: BaseHyperParams


@dataclass
class ExperimentConfig:
    experiment: BaseExperiment
