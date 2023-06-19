import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision

from data.dataset import FamilyHistoryDataSet
from models.models import CNN
from utils.optimizer import OptimizationLoop
from utils.training import BasicTraining
from utils.evaluation import MetricValidation

import hydra
from hydra.core.config_store import ConfigStore
from config import IsicConfig


cs = ConfigStore.instance()
cs.store(name='isic_config', node=IsicConfig)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: IsicConfig):
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
    EXPERIMENT_METADATA = cfg.benign_malignant_experiment.metadata
    EXPERIMENT_LABELS = cfg.benign_malignant_experiment.label_col
    DATA_COL = cfg.isic_paths.data_col

    # Mean & std for 85x85 cropped images
    IMG_CROP_SIZE = cfg.data_params.img_crop_size
    ISIC_MEAN = cfg.data_params.isic_mean
    ISIC_STD = cfg.data_params.isic_std

    dataset = FamilyHistoryDataSet(
        metadata_path=EXPERIMENT_METADATA,
        data_dir= ISIC_DATA_PATH,
        data_col=DATA_COL,
        ylabel_col=EXPERIMENT_LABELS,
        transforms=Compose([CenterCrop(IMG_CROP_SIZE),
                            ToTensor(),
                            Normalize(ISIC_MEAN, ISIC_STD)])
            )

    train_split, test_split = dataset.get_splits()
    train_set, test_set = random_split(dataset, [train_split, test_split])
    
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=n_workers)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=n_workers)

    optim_loop = OptimizationLoop(
        model=model,
        training=BasicTraining(nn.BCEWithLogitsLoss(),
                               optim.SGD(model.parameters(), lr=learning_rate)),
        validation=MetricValidation(),
        train_loader=train_loader,
        test_loader=test_loader,
        train_metrics=MetricCollection([Accuracy(task='binary'),
                                        AUROC(task='binary'),
                                        Precision(task='binary')]),
        test_metrics=MetricCollection([Accuracy(task='binary'),
                                       AUROC(task='binary'),
                                       Precision(task='binary')]),
        epochs=epochs,
        device=device
        )
    optim_loop.optimize()


if __name__ == '__main__':
    main()
