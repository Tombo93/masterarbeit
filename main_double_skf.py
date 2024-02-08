import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall, FBetaScore

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from data.dataset import FXDataset, Subset
from models.models import BatchNormCNN, ResNet, VGG, GoogleNet
from utils.optimizer import OptimizationLoop
from utils.training import PlotLossTraining
from utils.evaluation import MetricAndLossValidation
from utils.metrics import AverageMetricDict

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from config import ExperimentConfig


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
    # vgg_net = VGG(cfg.data_params.classes, finetuning=True)
    # vit_16 = VisionTransformer16(cfg.data_params.classes, finetuning=True)
    # google_net = GoogleNet(cfg.data_params.classes, finetuning=True)
    # batchnorm_net = BatchNormCNN(
    #     cfg.experiment.dataset.classes,
    #     cfg.experiment.dataset.channels,
    #     cfg.experiment.dataset.size,
    # )
    ### Class for averaging Metrics calculated on each fold ###
    avg_metrics = AverageMetricDict()

    mskf = MultilabelStratifiedKFold(n_splits=5)
    multi_label = np.concatenate(
        (
            np.expand_dims(data.labels, axis=1),
            np.expand_dims(data.extra_labels, axis=1),
        ),
        axis=1,
    )
    print(f"Training K-fold Cross Validation")
    for fold, (train_indices, val_indices) in enumerate(
        mskf.split(X=data.imgs, y=multi_label)
    ):
        train_set = Subset(dataset=data, indices=train_indices)
        val_set = Subset(dataset=data, indices=val_indices)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
        optim_loop = OptimizationLoop(
            model=model,
            training=PlotLossTraining(
                loss,
                optimizer_partial(model.parameters()),
            ),
            validation=MetricAndLossValidation(loss),
            train_loader=train_loader,
            test_loader=val_loader,
            train_metrics=MetricCollection(
                [
                    Recall(task="binary"),
                    Accuracy(task="binary"),
                    AUROC(task="binary"),
                    Precision(task="binary"),
                    FBetaScore(task="binary", beta=0.5),
                ]
            ).to(device),
            val_metrics=MetricCollection(
                [
                    Recall(task="binary"),
                    Accuracy(task="binary"),
                    AUROC(task="binary"),
                    Precision(task="binary"),
                    FBetaScore(task="binary", beta=0.5),
                ]
            ).to(device),
            epochs=epochs,
            device=device,
            logdir=f"logs/{filename}/{model.name}/{batch_size}/lr{lr}/fold_{fold}",
            kfold=True,
        )
        return
        optim_loop.optimize()
        ### Collect data per fold for later averaging ###
        train_metrics, val_metrics = optim_loop.get_fold_metrics()
        avg_metrics.add(train_dict=train_metrics, val_dict=val_metrics)

    avg_train_metrics, avg_val_metrics = avg_metrics.compute()
    df = pd.DataFrame(avg_train_metrics)
    df.to_csv(
        f"Multi-TrainMetrics{filename}-model-{model.name}-batchsize-{batch_size}-lr-{lr}.csv"
    )
    df = pd.DataFrame(avg_val_metrics)
    df.to_csv(
        f"Multi-ValMetrics{filename}-model-{model.name}-batchsize-{batch_size}-lr-{lr}.csv"
    )


if __name__ == "__main__":
    main()
