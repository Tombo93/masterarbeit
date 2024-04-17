from dataclasses import dataclass


@dataclass
class Data:
    data: str
    metadta: str
    classes: dict
    poison_class: str
    poison_encoding: int
    family_history: dict


@dataclass
class HParams:
    epochs: int
    lr: float
    batch_size: int
    num_workers: int
    rng_seed: int
    momentum: float
    decay: float


@dataclass
class Report:
    path: str
    test_report: str
    train_report: str


@dataclass
class Model:
    path: str


@dataclass
class PreProcessing:
    raw_metadata: str
    raw_data_dir: str
    interim_metadata: str
    interim_data: str
    backdoor_metadata: str
    backdoor_data: str
    benign_others: dict
    malignant_others: dict


@dataclass
class Config:
    data: Data
    hparams: HParams
    reports: Report
    model: Model
    preprocessing: PreProcessing
