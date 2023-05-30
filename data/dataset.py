import os
import torch
import pandas as pd
from PIL import Image


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