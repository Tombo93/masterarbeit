import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision

from pathlib import Path

from data.dataset import FamilyHistoryDataSet
from models.models import CNN
from utils.optimizer import OptimizationLoop
from utils.training import BasicTraining
from utils.evaluation import MetricValidation


import hydra


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    print(cfg)

    return
    # data
    ISIC_DATA_PATH = Path('data/ISIC/data').absolute()
    ISIC_YLABELS = Path('data/ISIC/family_history.csv').absolute()
    ISIC_METADATA = Path('data/ISIC/metadata_combined.csv').absolute()
    ISIC_ROOT_DIR = Path('data/ISIC').absolute()

    # Mean & std for 85x85 cropped images
    ISIC_MEAN = [1.2721, 0.3341, -0.0479]
    ISIC_STD = [0.2508, 0.2654, 0.3213]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparams
    learning_rate = 0.01
    batch_size = 32
    epochs = 100
    img_crop_size = 85
    n_classes = 1
    in_channels = 3
    n_workers = 4

    # Model
    model = CNN(n_classes, in_channels)
    # model = SimpleCNN(n_classes, in_channels)
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
