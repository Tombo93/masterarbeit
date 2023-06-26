import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision, Recall, ROC

from data.dataset import FamilyHistoryDataSet
from models.models import CNN
from utils.optimizer import OptimizationLoop
from utils.training import BasicTraining, PlotLossTraining
from utils.evaluation import MetricValidation, MetricAndLossValidation
from utils.metrics import AverageMeterCollection

import hydra
from hydra.core.config_store import ConfigStore
from config import IsicConfig


cs = ConfigStore.instance()
cs.store(name='isic_config', node=IsicConfig)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: IsicConfig):
    print(f'Experiment {cfg.family_history_experiment.label_col} parameters:')
    print(cfg.hyper_params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparams
    learning_rate = cfg.hyper_params.learning_rate
    batch_size = cfg.hyper_params.batch_size
    epochs = cfg.hyper_params.epochs
    n_workers = cfg.hyper_params.num_workers

    # Model
    n_classes = cfg.data_params.classes
    in_channels = cfg.data_params.channels 
    model = CNN(n_classes, in_channels)
    model.to(device)

    # data
    ISIC_DATA_PATH = cfg.isic_paths.isic_data_path
    EXPERIMENT_METADATA = cfg.family_history_experiment.metadata
    EXPERIMENT_LABELS = cfg.family_history_experiment.label_col
    DATA_COL = cfg.isic_paths.data_col

    # Mean & std for 85x85 cropped images
    IMG_CROP_SIZE = cfg.data_params.img_crop_size
    ISIC_MEAN = cfg.data_params.isic_mean
    ISIC_STD = cfg.data_params.isic_std

    # transforms = Compose([CenterCrop(IMG_CROP_SIZE), ToTensor(), Normalize(ISIC_MEAN, ISIC_STD)])
    transforms = Compose([CenterCrop(IMG_CROP_SIZE), ToTensor()])

    dataset = FamilyHistoryDataSet(
        metadata_path=EXPERIMENT_METADATA,
        data_dir= ISIC_DATA_PATH,
        data_col=DATA_COL,
        ylabel_col=EXPERIMENT_LABELS,
        transforms=transforms
        )

    train_split, test_split = dataset.get_splits()
    train_set, test_set = random_split(dataset, [train_split, test_split])
    
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=n_workers)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=n_workers)

    for learning_rate in [0.1, 0.01, 0.001]:
        optim_loop = OptimizationLoop(
        model=model,
        training= PlotLossTraining(nn.BCEWithLogitsLoss(),
                               optim.SGD(model.parameters(), lr=learning_rate)),
        validation=MetricAndLossValidation(nn.BCEWithLogitsLoss()),
        train_loader=train_loader,
        test_loader=test_loader,
        train_metrics=MetricCollection([Recall(task='binary', validate_args=True),
                                        Accuracy(task='binary', validate_args=True),
                                        AUROC(task='binary', validate_args=True),
                                        Precision(task='binary', validate_args=True)]),
        test_metrics=MetricCollection([Recall(task='binary', validate_args=True),
                                       Accuracy(task='binary', validate_args=True),
                                       AUROC(task='binary', validate_args=True),
                                       Precision(task='binary', validate_args=True)]),
        epochs=epochs,
        device=device
        )

        # optim_loop.optimize()
        optim_loop.overfit_batch_test(
            nn.BCEWithLogitsLoss(),
            optim.SGD(model.parameters(), lr=learning_rate),
            4,
            cfg.hyper_params.batch_size)


if __name__ == '__main__':
    main()
