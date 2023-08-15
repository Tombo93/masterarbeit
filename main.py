import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall

from sklearn.model_selection import StratifiedKFold

from data.dataset import FXDataset, Subset
from data.dataloader import FamilyHistoryDataloader
from models.models import BatchNormCNN, ResNet
from utils.optimizer import OptimizationLoop
from utils.training import PlotLossTraining
from utils.evaluation import MetricAndLossValidation
from utils.metrics import AverageMetricDict

import hydra
from hydra.core.config_store import ConfigStore
from config import IsicConfig

from data_preprocessing import get_my_indices


cs = ConfigStore.instance()
cs.store(name="isic_config", node=IsicConfig)

# logging.basicConfig(filename="kfold_indices.log", level=logging.info)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: IsicConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.hyper_params.epochs
    for filename in [
        # "20230710_ISIC_resize",
        # "20230712_ISIC_4000x6000_resize500x500",
        # "20230711_ISIC_4000x6000",
        "20230609_ISIC_85x85",
    ]:
        data = FXDataset(
            split="no_split",
            npz_folder="data/ISIC/",
            npz_file_name=filename,
            transforms=ToTensor(),
        )
        skf = StratifiedKFold(n_splits=5)
        lrs = [0.01, 0.001, 0.0001]
        batch_sizes = [32, 64]
        logger.info(f"Experiment")
        logger.info(f"Metadata")
        logger.info(f"----------")
        logger.info(
            f"Epochs: {epochs} | lrs: {lrs} | batch_sizes: {batch_sizes} | device: {device} | data used: {filename}"
        )
        logger.info(f"----------")

        resnet = ResNet(cfg.data_params.classes)
        batchnorm_net = BatchNormCNN(cfg.data_params.classes, cfg.data_params.channels)

        for learning_rate in lrs:
            for batch_size in batch_sizes:
                print(f"Training K-fold Cross Validation")

                ### Class for averaging Metrics calculated on each fold ###
                avg_metrics = AverageMetricDict()

                for fold, (train_indices, val_indices) in enumerate(
                    skf.split(data.imgs, data.labels)
                ):
                    logger.info(f"Fold {fold}")
                    logger.info(
                        f"train -  {np.bincount(data.labels[train_indices])}   |   test -  {np.bincount(data.labels[val_indices])}"
                    )
                    logger.info(f"lr: {learning_rate} | batch_size: {batch_size}")
                    logger.info(f"Indices of fold")
                    logger.info(f"Train")
                    logger.info(f"-----------------------------")
                    logger.info(f"{' '.join(map(str, train_indices))}")
                    logger.info(f"-----------------------------")
                    logger.info(f"Validation")
                    logger.info(f"-----------------------------")
                    logger.info(f"{' '.join(map(str, val_indices))}")
                    models = [copy.deepcopy(resnet)]
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
                            logdir=f"logs/{filename}/{model.name}/{batch_size}/lr{learning_rate}/fold_{fold}",
                            kfold=True,
                        )
                        optim_loop.optimize()

                        ### Collect data per fold for later averaging ###
                        train_metrics, val_metrics = optim_loop.get_fold_metrics()
                        avg_metrics.add(train_dict=train_metrics, val_dict=val_metrics)

                _, avg_val_metrics = avg_metrics.compute()
                fig, ax = plt.subplots(nrows=2, ncols=2)
                plt.subplots_adjust(hspace=0.5)
                for i, (k, v) in enumerate(avg_val_metrics.items()):
                    ax = plt.subplot(2, 2, i + 1)
                    # plt.figure(k)
                    ax.set_title(f"{k}")
                    ax.set_xlabel("epoch")
                    xs = range(0, len(v))
                    ys = v
                    ax.plot(xs, ys, "-.")
                    # plt.plot(xs, ys, "-.")
                    fig.savefig(
                        f"Validation Metrics Averaged file-{filename}-batchsize-{batch_size}-lr-{learning_rate}.png"
                    )


if __name__ == "__main__":
    main()
