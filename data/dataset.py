import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm


class FamilyHistoryDataSet(torch.utils.data.Dataset):
  """
  encoding: for multiclass, would you recommend a specific encoding?
  just numbers are fine
  
  """
  def __init__(self, metadata, root_dir, transforms=None, data_col=None, ylabel_col=None):
    self.root_dir = root_dir
    self.transforms = transforms

    self.annotations = pd.read_csv(metadata)

    self.xdata_col = self.annotations.columns.get_loc(data_col)
    self.ylabel_col = self.annotations.columns.get_loc(ylabel_col)

    self.class_encoding = {label : torch.tensor(idx) for idx, label in enumerate(self.annotations[ylabel_col].unique())}
    # self.class_encoding = lambda x : torch.tensor(int(x))
    self.encoding = F.one_hot(torch.arange(0, len(self.class_encoding)))

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, self.xdata_col] + '.JPG')
    image = Image.open(img_path)
    y_label = self.annotations.iloc[index, self.ylabel_col]
    # y_label = torch.tensor(int(self.annotations.iloc[index, self.ylabel_col]))
    if self.encoding is not None:
      y_label = self.encoding[self.class_encoding[y_label]]
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
