from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Data:
    id: str
    data: str
    metadata: str
    label_columns: Dict[str, str]
    classes: Dict[str, int]
    poison_class: str
    poison_encoding: int
    poison_whole_class: bool
    family_history: Dict[str, int]


@dataclass
class HParams:
    id: str
    epochs: int
    lr: float
    batch_size: int
    num_workers: int
    rng_seed: int
    momentum: float
    decay: float


@dataclass
class Model:
    path: Dict[str, str]


@dataclass
class Task:
    train: str
    test: str
    metrics: str
    num_classes: int
    reports: str


@dataclass
class PreProcessing:
    raw_metadata: str
    raw_data_dir: str
    interim_metadata: str
    interim_data: str
    backdoor_metadata: str
    backdoor_data: str
    benign_others: Dict[str, str]
    malignant_others: Dict[str, str]
    map_keys: Dict[str, List[str]]
    dropna_subset: List[str]
    drop_rows: Dict[str, str]
    poison_ratio: float


@dataclass
class Config:
    data: Data
    hparams: HParams
    preprocessing: PreProcessing
    task: Task
    model: Model
    backdoor: Dict[str, str]
