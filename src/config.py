from dataclasses import dataclass


@dataclass
class Data:
    data: str
    metadta: str
    classes: dict


@dataclass
class HParams:
    epochs: int
    lr: float
    batch_size: int
    num_workers: int
    rng_seed: int


@dataclass
class Report:
    path: str
    test_report: str
    train_report: str


@dataclass
class Config:
    data: Data
    hparams: HParams
    reports: Report
