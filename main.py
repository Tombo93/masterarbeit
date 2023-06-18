import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision

from data.dataset import FamilyHistoryDataSet, batch_mean_and_sd
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
    print(cfg)

    # data
    ISIC_DATA_PATH = cfg.isic_paths.isic_data_path
    ISIC_YLABELS = cfg.family_history_experiment.metadata
    ISIC_METADATA = cfg.isic_paths.isic_metadata
    ISIC_ROOT_DIR = cfg.isic_paths.isic_root_dir

    # Mean & std for 85x85 cropped images
    ISIC_MEAN = cfg.data_params.isic_mean
    ISIC_STD = cfg.data_params.isic_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparams
    learning_rate = cfg.hyper_params.learning_rate
    batch_size = cfg.hyper_params.batch_size
    epochs = cfg.hyper_params.epochs
    n_workers = cfg.hyper_params.num_workers

    img_crop_size = cfg.data_params.img_crop_size
    n_classes = cfg.data_params.classes
    in_channels = cfg.data_params.channels
 
    # Model
    model = CNN(n_classes, in_channels)
    model.to(device)

    """
    # prefer cropping images vs. resizing to not loose details
    # get minimum crop size that includes all images
    # check normalization
    """
    dataset = FamilyHistoryDataSet(
        metadata=ISIC_YLABELS,
        root_dir = ISIC_DATA_PATH,
        transforms=Compose(
            [CenterCrop(img_crop_size),
        	ToTensor(),
            Normalize(ISIC_MEAN, ISIC_STD)]
            ),
        data_col='isic_id',
        ylabel_col='family_hx_mm')

    train_split, test_split = dataset.get_splits()
    train_set, test_set = random_split(dataset, [train_split, test_split])
    
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=n_workers)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=n_workers)

    params = {
        'n_epochs': epochs,
        'train_loop': None,
        'validation_loop': None,
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'loss': nn.BCEWithLogitsLoss(), # nn.CrossEntropyLoss(),
        'optim': optim.SGD(model.parameters(), lr=learning_rate),
        'metrics' : {
            'train' : MetricCollection([
                Accuracy(task='binary'),
                AUROC(task='binary'),
                Precision(task='binary')]),
            'valid' :  MetricCollection([
                Accuracy(task='binary'),
                AUROC(task='binary'),
                Precision(task='binary')])
        },
        'logdir' : None,
        'device': device
    }
    optim_loop = OptimizationLoop(params, BasicTraining(), MetricValidation())
    optim_loop.optimize()


if __name__ == '__main__':
    main()
