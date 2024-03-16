import copy

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from config import ExperimentConfig

from data.dataset import FXDataset
from models.models import ResNet
from utils.optimizer import OptimizationLoop
from utils.training import PlotLossTraining
from utils.evaluation import MetricAndLossValidation
from utils.metrics import AverageMetricDict


cs = ConfigStore.instance()
cs.store(name="isic_config", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.experiment.hparams.epochs
    lr = cfg.experiment.hparams.lr
    batch_size = cfg.experiment.hparams.batch_size
    n_workers = cfg.experiment.hparams.num_workers
    shuffle = cfg.experiment.hparams.shuffle
    pin_memory = cfg.experiment.hparams.pin_memory
    loss = instantiate(cfg.experiment.hparams.loss)
    task = "binary"
    optimizer_partial = instantiate(cfg.experiment.hparams.optimizer)
    filename = cfg.experiment.dataset.name
    print(OmegaConf.to_yaml(cfg))

    data = FXDataset(
        split="no_split",
        npz_folder="data/ISIC/",
        npz_file_name=filename,
        transforms=ToTensor(),
    )
    resnet = ResNet(cfg.experiment.dataset.classes, finetuning=True)
    model = copy.deepcopy(resnet)
    model.to(device)

    avg_metrics = AverageMetricDict()

    is_multi_label_skf = False
    if is_multi_label_skf:
        mskf = MultilabelStratifiedKFold(n_splits=5)
        multi_label = np.concatenate(
            (
                np.expand_dims(data.labels, axis=1),
                np.expand_dims(data.extra_labels, axis=1),
            ),
            axis=1,
        )
        stratifier = mskf.split(X=data.imgs, y=multi_label)
    else:
        skf = StratifiedKFold(n_splits=5)
        stratifier = skf.split(X=data.imgs, y=data.labels)

    for fold, (train_indices, val_indices) in enumerate(stratifier):
        train_set = Subset(data, train_indices)
        val_set = Subset(data, val_indices)

        train_loader = DataLoader(train_set, batch_size, n_workers, shuffle, pin_memory)
        val_loader = DataLoader(val_set, batch_size, n_workers, shuffle, pin_memory)

        train_metrics = MetricCollection([Accuracy(task)]).to(device)
        val_metrics = MetricCollection([Accuracy(task)]).to(device)

        optim_loop = OptimizationLoop(
            model,
            PlotLossTraining(
                loss,
                optimizer_partial(model.parameters()),
            ),
            MetricAndLossValidation(loss),
            train_loader,
            val_loader,
            train_metrics,
            val_metrics,
            epochs,
            device,
            f"logs/{filename}/{model.name}/{batch_size}/lr{lr}/fold_{fold}",
            True,
        )
        optim_loop.optimize()

        train_metrics, val_metrics = optim_loop.get_fold_metrics()
        avg_metrics.add(train_dict=train_metrics, val_dict=val_metrics)

    avg_train_metrics, avg_val_metrics = avg_metrics.compute()
    df = pd.DataFrame(avg_train_metrics)
    df.to_csv(f"Multi-TrainMetrics{filename}-model-{model.name}-batchsize-{batch_size}-lr-{lr}.csv")
    df = pd.DataFrame(avg_val_metrics)
    df.to_csv(f"Multi-ValMetrics{filename}-model-{model.name}-batchsize-{batch_size}-lr-{lr}.csv")


if __name__ == "__main__":
    main()
