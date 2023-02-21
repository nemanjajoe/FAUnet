
import torch
import os

from network import FAUnet
from utils import DiceLoss

NCLS = 8
FAUnet_synapse= {
  'save_path' : 'D:\\Projects\\results\\FAUnet',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : 240,
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FAUnet,
  'model_args': {'img_size':224, 'in_chans':1, 'out_chans':NCLS},
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.00006, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':128, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy':2,
  'flags':{
    'save_history':False
  }
}

FAUnet_synapse_pretrain= {
  'save_path' : 'D:\\Projects\\results\\FAUnet\\synapes_pretain',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : int(1e5),
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FAUnet,
  'model_args': {'img_size':224, 'in_chans':1, 'out_chans':NCLS},
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.Adam,
  'optimizer_args': {'lr' : 1e-4},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy': 100,
  'flags' :{
    'save_history':True
  }
}

FAUnet_synapse_pretrain2= {
  'save_path' : 'D:\\Projects\\results\\FAUnet\\synapes_pretain2',
  'device'    : 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'epoch_num' : int(1e4),
  'num_classes': NCLS,
  'pretrained_params' : None,
  'model'     : FAUnet,
  'model_args': {'img_size':224, 'in_chans':1, 'out_chans':NCLS},
  'criterion' : DiceLoss,
  'criterion_args': {'n_classes' : NCLS}, # number of classes should equal to the number of out channels in model_args
  'optimizer' : torch.optim.AdamW,
  'optimizer_args': {'lr' : 0.0001, 'betas' : (0.9, 0.999), 'weight_decay' : 0.01},
  'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
  'scheduler_args':{},
  'train_loader_args' : {'batch_size':128, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'validate_loader_args': {'batch_size':64, 'shuffle':True, 'num_workers':1, 'pin_memory':True, 'drop_last':True},
  'eval_frequncy': 500,
  'flags' :{
    'save_history':True
  }
}

