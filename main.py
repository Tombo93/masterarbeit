import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC, Precision

from data import ISIC_DATA_PATH, ISIC_YLABELS, ISIC_MEAN, ISIC_STD, ISIC_METADATA
from data.dataset import FamilyHistoryDataSet
from models.models import SimpleCNN
from utils.evaluation import metrics_validation
from utils.training import OptimizationLoop, basic_training_loop


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
learning_rate = 0.01
batch_size = 32
epochs = 100
img_crop_size = 85
n_classes = 2
in_channels = 3

# Model
model = SimpleCNN(n_classes, in_channels)
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
train_set, test_set = torch.utils.data.random_split(dataset, [train_split, test_split])

train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=4)

params = {
    'n_epochs': epochs,
    'train_loop': basic_training_loop,
    'validation_loop': metrics_validation,
    'model': model,
    'train_loader': train_loader,
    'test_loader': test_loader,
    'loss': nn.BCELoss(), # nn.CrossEntropyLoss(),
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
optim_loop = OptimizationLoop(params)
optim_loop.optimize()
