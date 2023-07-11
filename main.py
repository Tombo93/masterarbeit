import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall

from sklearn.model_selection import StratifiedKFold

from data.dataset import FXDataset, Subset
from models.models import BatchNormCNN, ResNet
from utils.optimizer import OptimizationLoop
from utils.training import PlotLossTraining
from utils.evaluation import MetricAndLossValidation

import hydra
from hydra.core.config_store import ConfigStore
from config import IsicConfig


cs = ConfigStore.instance()
cs.store(name="isic_config", node=IsicConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: IsicConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.hyper_params.epochs

    data = FXDataset(
        split="no_split",
        npz_folder="data/ISIC/",
        npz_file_name="20230711_ISIC_4000x6000",  # "20230710_ISIC_resize",
        transforms=ToTensor(),
    )
    skf = StratifiedKFold(n_splits=5)
    lrs = [0.01]
    batch_sizes = [32]
    for learning_rate in lrs:
        for batch_size in batch_sizes:
            print(f"Training K-fold Cross Validation")
            for fold, (train_indices, val_indices) in enumerate(
                skf.split(data.imgs, data.labels)
            ):
                print([f"Fold {fold}"])
                print(
                    f"train -  {np.bincount(data.labels[train_indices])}   |   test -  {np.bincount(data.labels[val_indices])}"
                )
                models = [
                    ResNet(cfg.data_params.classes),
                    # BatchNormCNN(cfg.data_params.classes, cfg.data_params.channels),
                ]
                for model in models:
                    model.to(device)

                    train_set = Subset(dataset=data, indices=train_indices)
                    val_set = Subset(dataset=data, indices=val_indices)
                    train_loader = DataLoader(
                        train_set,
                        batch_size=batch_size,
                        num_workers=cfg.hyper_params.num_workers,
                        shuffle=True,
                        pin_memory=True,
                    )
                    val_loader = DataLoader(
                        val_set,
                        batch_size=batch_size,
                        num_workers=cfg.hyper_params.num_workers,
                        shuffle=True,
                        pin_memory=True,
                    )

                    optim_loop = OptimizationLoop(
                        model=model,
                        training=PlotLossTraining(
                            nn.BCEWithLogitsLoss(),
                            optim.SGD(model.parameters(), lr=learning_rate),
                        ),
                        validation=MetricAndLossValidation(nn.BCEWithLogitsLoss()),
                        train_loader=train_loader,
                        test_loader=val_loader,
                        train_metrics=MetricCollection(
                            [
                                Recall(task="binary"),
                                Accuracy(task="binary"),
                                AUROC(task="binary"),
                                Precision(task="binary"),
                            ]
                        ).to(device),
                        val_metrics=MetricCollection(
                            [
                                Recall(task="binary"),
                                Accuracy(task="binary"),
                                AUROC(task="binary"),
                                Precision(task="binary"),
                            ]
                        ).to(device),
                        epochs=epochs,
                        device=device,
                        logdir=f"runs/4000x6000/{model.name}/{batch_size}/lr{learning_rate}",
                    )
                    optim_loop.optimize()


if __name__ == "__main__":
    main()
