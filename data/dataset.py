import os
import torch
import pandas as pd
import numpy as np
from PIL import Image


class FamilyHistoryDataSet(torch.utils.data.Dataset):
  def __init__(self, csv_file, root_dir, transforms=None):
    self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
    self.root_dir = root_dir
    self.transforms = transforms

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    image = np.array(Image.open(img_path), copy=False, dtype='uint8') # das war das Problem?? kein Tensor?
    y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

    if self.transforms:
      image = self.transforms(image)

    return (image, y_label)

  def get_splits(self, splits=[0.8, 0.2]):
    train_split = round(len(self.annotations)*splits[0])
    test_split = len(self.annotations) - train_split
    return (train_split, test_split)