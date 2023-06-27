import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall
from medmnist import PneumoniaMNIST

from data.create_npz import CreateNpz
from data.dataloader import FamilyHistoryDataloader, MedMnistDataloader, FXNpzDataloader
from models.models import CNN, BatchNormCNN, ResNet
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
    print(f"Experiment {cfg.family_history_experiment.label_col} parameters:")
    print(cfg.hyper_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    learning_rate = cfg.hyper_params.learning_rate
    epochs = cfg.hyper_params.epochs

    # Model
    # model = BatchNormCNN(cfg.data_params.classes, cfg.data_params.channels)
    model = ResNet(cfg.data_params.classes)
    model.to(device)

    # Mean & std for 85x85 cropped images
    IMG_CROP_SIZE = cfg.data_params.img_crop_size
    ISIC_MEAN = cfg.data_params.isic_mean
    ISIC_STD = cfg.data_params.isic_std

    # fx_data = FamilyHistoryDataloader(
    #     metadata=cfg.family_history_experiment.metadata,
    #     datapath=cfg.isic_paths.isic_data_path,
    #     data_col=cfg.isic_paths.data_col,
    #     labels=cfg.family_history_experiment.label_col,
    #     transforms=Compose(
    #         [CenterCrop(IMG_CROP_SIZE), ToTensor(), Normalize(ISIC_MEAN, ISIC_STD)]
    #     ),
    #     batch_size=cfg.hyper_params.batch_size,
    #     num_workers=cfg.hyper_params.num_workers,
    # )
    # train_loader, test_loader = fx_data.get_dataloaders()
    # train_loader, test_loader, _ = MedMnistDataloader(
    #     PneumoniaMNIST, ToTensor()
    # ).get_medmnist_dataloaders()
    data = FXNpzDataloader(transforms=ToTensor())
    train_loader, test_loader = data.get_dataloaders()

    optim_loop = OptimizationLoop(
        model=model,
        training=PlotLossTraining(
            nn.BCEWithLogitsLoss(), optim.SGD(model.parameters(), lr=learning_rate)
        ),
        validation=MetricAndLossValidation(nn.BCEWithLogitsLoss()),
        train_loader=train_loader,
        test_loader=test_loader,
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
    )
    optim_loop.optimize()
    # optim_loop.overfit_batch_test(
    #     nn.BCEWithLogitsLoss(),
    #     optim.SGD(model.parameters(), lr=learning_rate),
    #     4,
    #     cfg.hyper_params.batch_size)


if __name__ == "__main__":
    main()
