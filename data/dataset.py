import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm


class FamilyHistoryDataSet(torch.utils.data.Dataset):
  def __init__(self, ylabels, root_dir, transforms=None):
    self.annotations = pd.read_csv(ylabels)
    self.root_dir = root_dir
    self.transforms = transforms

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    image = Image.open(img_path)
    y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

    if self.transforms:
      image = self.transforms(image)

    return (image, y_label)

  def get_splits(self, splits=[0.8, 0.2]):
    train_split = round(len(self.annotations)*splits[0])
    test_split = len(self.annotations) - train_split
    return (train_split, test_split)
  
  def get_imgs_lowest_width_height(self) -> int:
    '''Computes the smalles width/heigt of the images of the dataset'''
    height, width = float('inf'), float('inf')
    for index in range(len(self.annotations)):
      img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
      image = Image.open(img_path)
      if image.width < width:
        width = image.width
      if image.height < height:
        height = image.height
    return width, height


def get_mean_std(dataloader):
  '''Computes the mean and std of given dataset using a dataloader'''
  mean = 0.0
  for images, _ in tqdm(dataloader, leave=False):
      batch_samples = images.size(0) 
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
  mean = mean / len(dataloader.dataset)
  print(mean)
  var = 0.0
  for images, _ in tqdm(dataloader, leave=False):
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      var += ((images - mean.unsqueeze(1))**2).sum([0,2])
  std = torch.sqrt(var / (len(dataloader.dataset)*224*224))
  print(std)
  return mean, std
