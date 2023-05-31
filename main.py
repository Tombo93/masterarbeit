import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

from data import ISIC_DATA_PATH, ISIC_YLABELS, ISIC_MEAN, ISIC_STD
from data.dataset import FamilyHistoryDataSet, get_mean_std
from models.models import CNN
from utils.evaluation import check_accuracy
from utils.training import basic_training_loop


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
learning_rate = 1e-3
batch_size = 32
epochs = 1
img_crop_size = 85
n_classes = 4
in_features = 3
model = CNN(n_classes, in_features)

"""
# prefer cropping images vs. resizing to not loose details
# get minimum crop size that includes all images
# check normalization
"""
dataset = FamilyHistoryDataSet(
    ylabels=ISIC_YLABELS,
    root_dir = ISIC_DATA_PATH,
    transforms=Compose(
        [CenterCrop(img_crop_size),
    	ToTensor(),
        Normalize(ISIC_MEAN, ISIC_STD)]
        )
        )
# img_crop_size = dataset.get_imgs_lowest_width_height()

train_split, test_split = dataset.get_splits()
train_set, test_set = torch.utils.data.random_split(dataset, [train_split, test_split])

# all_loader = torch.utils.data.DataLoader(
#     dataset=dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=batch_size, shuffle=True)


# if not model:
#     model = torchvision.models.googlenet(pretrained=True)
#     model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# better not, better slow training and better models
# single_batch_test(epochs, train_loader, model, criterion, optimizer, device)


basic_training_loop(epochs, train_loader, model, criterion, optimizer, device)
print('Finished Training')

# print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model, device)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model, device)
