import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from data import ISIC_DATA_PATH, ISIC_YLABELS
from data.dataset import FamilyHistoryDataSet
from utils.evaluation import check_accuracy
from utils.training import basic_training_loop, single_batch_test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
learning_rate = 1e-3
batch_size = 4
epochs = 1
img_size = 300
model = False


dataset = FamilyHistoryDataSet(ylabels=ISIC_YLABELS,
                               root_dir = ISIC_DATA_PATH,
                               transforms=transforms.Compose(
                                   [transforms.Resize((img_size)), transforms.ToTensor()] # prefer cropping imgs vs. resizing + get minimum crop that includes all imgs
                                   ) # check normalization
                                   )

train_split, test_split = dataset.get_splits()
train_set, test_set = torch.utils.data.random_split(dataset, [train_split, test_split])

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

if not model:
    model = torchvision.models.googlenet(pretrained=True)
    model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# better not, better slow training and better models
single_batch_test(epochs, train_loader, model, criterion, optimizer, device)

'''
basic_training_loop(epochs, train_loader, model, criterion, optimizer, device)
print('Finished Training')

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model, device)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model, device)'''
