# Dataset class FOR INFERENCE

# 1. MFCC
# 2. STFT
# 3. MRCG


import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import preprocessing
import MRCG
# import sys

SR = 44100

#=============================   MFCC 1D  =================================#
class TEST_MFCC_1D(Dataset):
    def __init__(self, csv_path, file_path):
        metaData = pd.read_csv(csv_path)
        metaData = metaData.sort_values(by=['filename'])
        self.file_names = []
        self.file_path = file_path

        for i in range(0, len(metaData)):
            self.file_names.append(metaData.iloc[i, 0])

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mfcc = librosa.feature.mfcc(y = data, sr = SR, hop_length = 1024,
                            n_mfcc = 40)

        soundFormatted = torch.from_numpy(mfcc).float()
    
        return soundFormatted, self.file_names[index]

    def __len__(self):
        return len(self.file_names)


#=============================   MFCC 2D  =================================#
class TEST_MFCC_2D(Dataset):
    def __init__(self, csv_path, file_path):
        metaData = pd.read_csv(csv_path)
        metaData = metaData.sort_values(by=['filename'])
        self.file_names = []
        self.file_path = file_path

        for i in range(0, len(metaData)):
            self.file_names.append(metaData.iloc[i, 0])

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mfcc = librosa.feature.mfcc(y = data, sr = SR, hop_length = 1024,
                            n_mfcc = 40)
        
        soundFormatted = torch.from_numpy(mfcc).float()
        soundFormatted = torch.unsqueeze(soundFormatted, dim=0)
        soundFormatted = torch.cat((soundFormatted,soundFormatted,soundFormatted))

        return soundFormatted, self.file_names[index]
  
    def __len__(self):
        return len(self.file_names)


#=============================   MRCG 1D  =================================#

class TEST_MRCG_1D(Dataset):

    def __init__(self, csv_path, file_path):
        metaData = pd.read_csv(csv_path)
        metaData = metaData.sort_values(by=['filename'])
        self.file_names = []
        self.file_path = file_path

        for i in range(0, len(metaData)):
            self.file_names.append(metaData.iloc[i, 0])

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mrcg = MRCG.mrcg_extract(data, SR)

        soundFormatted = torch.from_numpy(mrcg).float()
        # soundFormatted.transpose_(0, 1)
        # print("data shape", soundFormatted.shape)     
        return soundFormatted, self.file_names[index]
  
    def __len__(self):
        return len(self.file_names)

#=============================   MRCG 2D  =================================#

class TEST_MRCG_2D(Dataset):

    def __init__(self, csv_path, file_path):
        metaData = pd.read_csv(csv_path)
        metaData = metaData.sort_values(by=['filename'])
        self.file_names = []
        self.file_path = file_path

        for i in range(0, len(metaData)):
            self.file_names.append(metaData.iloc[i, 0])

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mrcg = MRCG.mrcg_extract(data, SR)

        soundFormatted = torch.from_numpy(mrcg).float()

        # soundFormatted.transpose_(0, 1)
        soundFormatted = soundFormatted.unsqueeze(dim=0)
        print("mrgc shape", soundFormatted.shape)     
        return soundFormatted, self.file_names[index]
  
    def __len__(self):
        return len(self.file_names)