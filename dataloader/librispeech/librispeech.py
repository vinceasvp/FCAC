import os
import os.path as osp
import re
import json
import time
import h5py
from matplotlib.font_manager import json_dump
import numpy as np
import random
import librosa
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
import pandas as pd
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class LBRS(Dataset):

    def __init__(self, root='./', phase='train', 
                 index_path=None, index=None, k=5, base_sess=None, data_type='audio', args=None):
        self.root = os.path.expanduser(root)
        self.root = root
        self.data_type = data_type
        # self.make_extractor()
        self.phase = phase
        # self.train = train  # training set or test set
        self.all_train_df = pd.read_csv(os.path.join(root, "librispeech_fscil_train.csv"))
        self.all_val_df = pd.read_csv(os.path.join(root, "librispeech_fscil_val.csv"))
        self.all_test_df = pd.read_csv(os.path.join(root, "librispeech_fscil_test.csv"))

        if phase == 'train':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_train_df, index, per_num=None)
        elif phase == 'val':
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.all_val_df, index, per_num=None)
            else:
                self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=k)
        elif phase =='test':
            self.data, self.targets = self.SelectfromClasses(self.all_test_df, index, per_num=None)
        


    def SelectfromClasses(self, df, index, per_num=None):
        data_tmp = []
        targets_tmp = []

        for i in index:
            ind_cl = np.where(i == df['label'])[0]
            # random.shuffle(ind_cl)

            k = 0
            # ind_cl is the index list whose label equals i, 
            # start_idx make sure 
            # there is no intersection between train and test set  
            for j in ind_cl:
                filename = os.path.join(df['filename'][j])
                path = os.path.join(self.root, filename)
                data_tmp.append(path)
                targets_tmp.append(df['label'][j])
                k += 1
                if per_num is not None:
                    if k >= per_num:
                        break
              
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        audio,sr = torchaudio.load(path)
        return audio.squeeze(0), targets

if __name__ == '__main__':

    # class_index = open(txt_path).read().splitlines()
    base_class = 60
    class_index = np.arange(base_class, 100)
    dataroot = "/data/datasets/librispeech_fscil/spk_segments"
    batch_size_base = 400
    trainset = LBRS(root=dataroot, phase="train",  index=class_index, k=5,
                      base_sess=False)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    list(trainloader)    
