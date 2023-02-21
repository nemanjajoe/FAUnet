# ------------------------------------------
# MRTNet: Mask Refinement Transformer Network
# Licensed under the MIT License.
# written By Ruixin Yang
# ------------------------------------------

import math
import os
import random
import numpy as np
import torch
import time

from torch.utils.data import DataLoader
from dataset import RandomTransform, Synapse_dataset

def train_epoch(model, criterion, optimizer, train_list, train_loader_args, device):
    model.train()
    running_loss = 0.0
    dataset = Synapse_dataset(train_list, RandomTransform((224,224)))
    data_loader = DataLoader(dataset, **train_loader_args)
    length = len(data_loader)

    for idx, (x, y) in enumerate(data_loader):
      x = x.to(device)
      y = y.to(device)

      y_ = model(x)

      loss = criterion(y_, y)
      running_loss += loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    return running_loss/length

def validate_epoch(model, criterion, validate_list, validate_loader_args,num_classes, device, scheduler=None):

    dataset = Synapse_dataset(validate_list, RandomTransform((224,224)))
    data_loader = DataLoader(dataset, **validate_loader_args)
    running_loss, dice_class = 0, torch.zeros(num_classes)
    length = len(data_loader)

    with torch.no_grad():
      model.eval()
      
      for idx, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        y_ = model(x)

        loss = criterion(y_, y)
          
        running_loss += loss
        dice_class += criterion.get_dice_class()

      if scheduler is not None:
          scheduler.step(loss)
          lr = scheduler.get_lr()
          print(f'\t lr : {lr}')

    return running_loss/length, dice_class/length

class Trainer_synapse():
  def __init__(self, dataset_path:str, hyper):
    super().__init__()
    #---------- setup model ----------
    print('Synapse Trainer Initailizing...')
    self.num_classes = hyper['num_classes']
    self.device = torch.device(hyper['device'])
    self.model = hyper['model'](**hyper['model_args']).to(self.device)
    if hyper['pretrained_params'] is not None:
      self.model.load_from(hyper['pretrained_params'], self.device)
      self.model.to(self.device)

    self.criterion = hyper['criterion'](**hyper['criterion_args'])
    self.optimizer = hyper['optimizer'](self.model.parameters(),**hyper['optimizer_args'])
    self.scheduler = hyper['scheduler'](self.optimizer,**hyper['scheduler_args']) if 'scheduler' in hyper else None
    self.epoch_num = hyper['epoch_num']
    self.dataset_base_dir = dataset_path

    #---------- random select train set and validate set ----------
    
    images_path = os.path.join(self.dataset_base_dir, 'train/images')
    label_path = os.path.join(self.dataset_base_dir, 'train/labels')
    names = [(os.path.join(images_path, name), os.path.join(label_path, name)) 
              for name in os.listdir(os.path.join(dataset_path, 'train/images/'))]
    random.shuffle(names)
    self.train_list = names[:math.floor(len(names)*0.8)]
    self.validate_list = names[math.floor(len(names)*0.8):]
  
    #---------- get dataloader args ----------
    self.train_loader_args = hyper['train_loader_args'] if 'train_loader_args' in hyper else {}
    self.validate_loader_args = hyper['validate_loader_args'] if 'validate_loader_args' in hyper else {}

    #---------- other configurations ----------
    self.eval_frequncy = hyper['eval_frequncy'] or 2
    self.save_path = hyper['save_path']
    self.save_epoch_path = os.path.join(self.save_path, 'best_epoch.pth')
    self.labels = {0: 'background',  1: 'spleen',      2: 'right kidney',
                   3: 'left kidney', 4: 'gallbladder', 5: 'liver',
                   6: 'stomach',     7: 'aorta',       8: 'pancreas'}

    if os.path.exists(self.save_path) == False:
      os.mkdir(self.save_path)

    if hyper['flags']['save_history']:
        self.save_history = True
        self.save_history_file_path = os.path.join(self.save_path,"history.txt")
        if os.path.exists(self.save_history_file_path):
            date = time.strftime("%Y%m%d_%H%M%S",time.localtime(time.time()))
            os.rename(self.save_history_file_path,f"history_{date}.txt")  

    print('Synapse Trainer initalied !')

  def test(self):
    train_epoch(self.model,self.criterion,self.optimizer,self.train_list,self.train_loader_args,self.device)
    print('train epoch success !')
    validate_epoch(self.model,self.criterion,self.validate_list,self.validate_loader_args,self.num_classes,self.device,self.scheduler)
    print('validate epoch success !')

  def train(self, continue_train=False, test_epoch=None):

    labels = [label for label in self.labels.values()][:self.num_classes]
    epoch_num = self.epoch_num
    if test_epoch is not None:
      epoch_num = test_epoch

    epoch_start = 0
    dice_list = [0] * self.num_classes

    if continue_train:
      params = torch.load(self.save_epoch_path)
      dice_list = params['dice_list']
      epoch_start = params['epoch']
      self.train_list = params['train_samples']
      self.validate_list = params['validate_samples']
      print(f'continue trainning from epoch {epoch_start}')
      for i, organ in enumerate(labels):
        print(f'\t {organ} : {dice_list[i]}')

      self.model.load_state_dict(params['state_dict'])
      self.optimizer.load_state_dict(params['optimizer_state_dict'])
    
    print("---------- start trainning ----------")
    print(f'train set: {len(self.train_list)}')
    print(f'validate set: {len(self.validate_list)}')
    for epoch in range(epoch_start, epoch_num):
        t_tick = time.time()
        print(f'epoch {epoch}/{epoch_num}:')
        loss = train_epoch(self.model,self.criterion,self.optimizer,self.train_list,self.train_loader_args,self.device)
        print(f'\t train loss: {loss.item():.4f}')
        print(f'\t time: {(time.time() - t_tick):.2f} s')


        if self.save_history:
          with open(self.save_history_file_path,'a') as f:
              f.write(f'epoch:{epoch}; loss:{loss.item():.3f}\n')

        if epoch > 0 and epoch % self.eval_frequncy == 0:
          val_loss, dice_ = validate_epoch(self.model,self.criterion,self.validate_list,self.validate_loader_args,self.num_classes,self.device,self.scheduler)
          dice_ = list(np.asarray(dice_))
          if np.average(dice_) > np.average(dice_list):
            params = {
              'epoch' : epoch,
              'state_dict' : self.model.state_dict(),
              'optimizer_state_dict' : self.optimizer.state_dict(),
              'loss' : val_loss.item(),
              'dice_list' : dice_,
              'labels'    : labels,
              'train_samples' : self.train_list,
              'validate_samples': self.validate_list
            }
            torch.save(params, self.save_epoch_path)

          print(f'\t validate loss: {val_loss.item():.4f}')
          print(f'\t validate dice:')
          for i, organ in enumerate(labels):
            print(f'\t\t {organ} : {dice_[i]:.4f}')

