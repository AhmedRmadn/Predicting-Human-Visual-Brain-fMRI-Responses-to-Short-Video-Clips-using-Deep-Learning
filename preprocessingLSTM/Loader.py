from dataset import VidFMRIDataset
from torch.utils.data import DataLoader
import os
import numpy as np

import torch
import timm
import tqdm
import glob
import pickle

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di

# train_dataset = VidFMRIDataset(train_vid_files, train_fmri_data, fmri_transform="combination_augment")
# train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
def test():
  vid_dir = "resnet152/"
  fmri_dir = "participants_data_v2021/mini_track/sub01/V1.pkl"
  video_list = glob.glob(vid_dir + '/*_out_2.npy')
  video_list.sort()
  video_list = np.asarray(video_list[:1000])
  print(video_list.shape)
  x = load_dict(fmri_dir)
  print(x['train'].shape)
  train_dataset = VidFMRIDataset(video_list,x['train'], fmri_transform="combination_augment")
  train_loader = DataLoader(train_dataset, num_workers=8, batch_size=2, shuffle=True)

  t = (train_loader)
  for i, sample in enumerate(t):
      print(i)
      print(sample['vid_data'].shape)
     


if __name__ == '__main__':
    test()