import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torchvision.transforms import Compose

from typing import Tuple, Union, Any
from torch.utils.data import DataLoader


class FamilyHistoryDataSet(torch.utils.data.Dataset):
  def __init__(self,
               metadata_path: str,
               data_dir: str,
               data_col: str,
               ylabel_col: str,
               transforms: Compose = None) -> None:
    self.data_dir = data_dir
    self.transforms = transforms
    self.annotations = pd.read_csv(metadata_path)
    self.xdata_col = self.annotations.columns.get_loc(data_col)
    self.ylabel_col = self.annotations.columns.get_loc(ylabel_col)

  def __len__(self) -> int:
    return len(self.annotations)

  def __getitem__(self, index) -> Tuple[torch.TensorType, torch.TensorType]:
    img_path = os.path.join(self.data_dir, self.annotations.iloc[index, self.xdata_col])
    image = Image.open(img_path)
    y_label = torch.tensor(int(self.annotations.iloc[index, self.ylabel_col]))
    if self.transforms is not None:
      image = self.transforms(image)
    return (image, y_label)

  def get_splits(self, splits=[0.8, 0.2]) -> Tuple[int, int]:
    train_split = round(len(self.annotations)*splits[0])
    test_split = len(self.annotations) - train_split
    return (train_split, test_split)
  
  def get_imgs_lowest_width_height(self) -> Tuple[Union[int, float], Union[int, float]]:
    '''Computes the smalles width/heigt of the images of the dataset'''
    height, width = float('inf'), float('inf')
    for index in range(len(self.annotations)):
      img_path = os.path.join(self.data_dir, self.annotations.iloc[index, 0])
      image = Image.open(img_path)
      if image.width < width:
        width = image.width
      if image.height < height:
        height = image.height
    return width, height


def get_mean_std(dataloader: DataLoader) -> Tuple[Union[float, Any], torch.TensorType]:
  '''Computes the mean and std of given dataset using a dataloader'''
  mean = 0.0
  for images, _ in dataloader:
      batch_samples = images.size(0) 
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
  mean = mean / len(dataloader.dataset)
  var = 0.0
  for images, _ in dataloader:
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      var += ((images - mean.unsqueeze(1))**2).sum([0,2])
  std = torch.sqrt(var / (len(dataloader.dataset)*224*224))
  return mean, std


def batch_mean_and_sd(loader: DataLoader) -> Tuple[torch.TensorType, torch.TensorType]:
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tqdm(loader, leave=False):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std