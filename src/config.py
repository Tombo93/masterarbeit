from dataclasses import dataclass
from typing import List
import torch


######################### Main config object #########################
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
