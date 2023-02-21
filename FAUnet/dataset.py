# ------------------------------------------
# MRTNet: Mask Refinement Transformer Network
# Licensed under the MIT License.
# written By Ruixin Yang
# ------------------------------------------

import os
import torch
import random
import numpy as np

from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data.dataset import Dataset


def random_rot_flip(image, label):
  k = np.random.randint(0, 4)
  image = np.rot90(image, k)
  label = np.rot90(label, k)
  axis = np.random.randint(0, 2)
  image = np.flip(image, axis=axis).copy()
  label = np.flip(label, axis=axis).copy()
  return image, label


def random_rotate(image, label):
  angle = np.random.randint(-20, 20)
  image = ndimage.rotate(image, angle, order=0, reshape=False)
  label = ndimage.rotate(label, angle, order=0, reshape=False)
  return image, label


class RandomTransform(object):
  def __init__(self, output_size):
    self.output_size = output_size

  def __call__(self, sample):
    image, label = sample
    if random.random() > 0.5:
      image, label = random_rot_flip(image, label)
    elif random.random() > 0.5:
      image, label = random_rotate(image, label)
    
    x, y = image.shape

    if x != self.output_size[0] or y != self.output_size[1]:
      image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
      label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    label = torch.from_numpy(label.astype(np.float32))

    return (image, label.long())


class Synapse_dataset(Dataset):
  def __init__(self, sample_list, transform=None, shuffle=True):
    self.transform = transform  # using transform in torch!
    self.sample_list = sample_list
    
    if shuffle:
      random.shuffle(self.sample_list)

  def __len__(self):
    return len(self.sample_list)
  
  def __getitem__(self, idx):
    x_path, y_path = self.sample_list[idx]
    x = np.load(x_path) 
    y = np.load(y_path)
    
    if self.transform:
      x, y = self.transform((x,y))
    else:
      x = torch.from_numpy(x)
      y = torch.from_numpy(y)

    return (x,y)