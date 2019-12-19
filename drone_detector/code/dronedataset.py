# Dataset class

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
class MFCC_1D(Dataset):
    def __init__(self, csv_path, file_path, folderList, tv, transform=None):
        metaData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []   
        self.transform = transform


        for i in range(0, len(metaData)):
            if metaData.iloc[i, 2] in folderList and metaData.iloc[i, 3] in tv:
            # if metaData.iloc[i, 2] in folderList:
                self.file_names.append(metaData.iloc[i, 0])
                self.labels.append(metaData.iloc[i, 1])
                # self.folders.append(metaData.iloc[i, 2])
                # self.tv_list.append(metaData.iloc[i, 3])
          
        self.file_path = file_path
        self.folderList = folderList
        self.tv = tv
    

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mfcc = librosa.feature.mfcc(y = data, sr = SR, hop_length = 1024,
                            n_mfcc = 40)

        if self.transform: 
            
            mfcc = self.transform(mfcc)
            soundFormatted = mfcc.float()
            soundFormatted = torch.squeeze(soundFormatted)   
            # soundFormatted.transpose_(0, 1) 
            # print("data shape", soundFormatted.shape)        
            return soundFormatted, self.labels[index]

        soundFormatted = torch.from_numpy(mfcc).float()
        # soundFormatted.transpose_(0, 1)
        # print("data shape", soundFormatted.shape)     
        return soundFormatted, self.labels[index]
  
    def __len__(self):
        return len(self.file_names)


#=============================   MFCC 2D  =================================#
class MFCC_2D(Dataset):
    def __init__(self, csv_path, file_path, folderList, tv, transform=None):
        metaData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []   
        self.tv_list = []
        self.transform = transform


        for i in range(0, len(metaData)):
            if metaData.iloc[i, 2] in folderList and metaData.iloc[i, 3] in tv:
            # if metaData.iloc[i, 2] in folderList:
                self.file_names.append(metaData.iloc[i, 0])
                self.labels.append(metaData.iloc[i, 1])
                # self.folders.append(metaData.iloc[i, 2])
                # self.tv_list.append(metaData.iloc[i, 3])
          
        self.file_path = file_path
        self.folderList = folderList
        self.tv = tv
    

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mfcc = librosa.feature.mfcc(y = data, sr = SR, hop_length = 1024,
                            n_mfcc = 40)

        if self.transform:
            
            mfcc = self.transform(mfcc)
            soundFormatted = mfcc.float()
            soundFormatted = torch.cat((soundFormatted,soundFormatted,soundFormatted)) #?? size 확인하기
            # print("Brinda", soundFormatted.shape)    
            return soundFormatted, self.labels[index]
        #print("soundformat shape:", soundFormatted.shape)
        
        soundFormatted = torch.from_numpy(mfcc).float()
        soundFormatted = torch.cat((soundFormatted,soundFormatted,soundFormatted)) #?? size 확인하기
#         soundFormatted = torch.unsqueeze(soundFormatted, dim=0)
#         soundFormatted = torch.cat((soundFormatted,soundFormatted,soundFormatted))

        return soundFormatted, self.labels[index]
  
    def __len__(self):
        return len(self.file_names)


#=============================   MRCG 1D  =================================#

class MRCG_1D(Dataset):

    def __init__(self, csv_path, file_path, folderList, tv, transform=None):
        metaData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []   
        self.transform = transform


        for i in range(0, len(metaData)):
            if metaData.iloc[i, 2] in folderList and metaData.iloc[i, 3] in tv:
            # if metaData.iloc[i, 2] in folderList:
                self.file_names.append(metaData.iloc[i, 0])
                self.labels.append(metaData.iloc[i, 1])
                # self.folders.append(metaData.iloc[i, 2])
                # self.tv_list.append(metaData.iloc[i, 3])
          
        self.file_path = file_path
        self.folderList = folderList
        self.tv = tv
    

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mrcg = MRCG.mrcg_extract(data, SR)

        if self.transform: 
            
            mrcg = self.transform(mrcg)
            soundFormatted = mrcg.float()
            soundFormatted = torch.squeeze(soundFormatted)
            # soundFormatted.transpose_(0, 1)  
            print("data shape", soundFormatted.shape)     
            return soundFormatted, self.labels[index]

        soundFormatted = torch.from_numpy(mrcg).float()
        # soundFormatted.transpose_(0, 1)
        print("data shape", soundFormatted.shape)     
        return soundFormatted, self.labels[index]
  
    def __len__(self):
        return len(self.file_names)

#=============================   MRCG 2D  =================================#

class MRCG_2D(Dataset):

    def __init__(self, csv_path, file_path, folderList, tv, transform=None):
        metaData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []   
        self.transform = transform


        for i in range(0, len(metaData)):
            if metaData.iloc[i, 2] in folderList and metaData.iloc[i, 3] in tv:
            # if metaData.iloc[i, 2] in folderList:
                self.file_names.append(metaData.iloc[i, 0])
                self.labels.append(metaData.iloc[i, 1])
                # self.folders.append(metaData.iloc[i, 2])
                # self.tv_list.append(metaData.iloc[i, 3])
          
        self.file_path = file_path
        self.folderList = folderList
        self.tv = tv
    

    def __getitem__(self, index):
        
        path = self.file_path + '/' + self.file_names[index] + '.wav'
        data = librosa.core.load(path, sr = SR, mono = True)[0]

        #pre-emphasis
        data = preprocessing.preemphasis(data)

        mrcg = MRCG.mrcg_extract(data, SR)

        # if self.transform: 
            
        #     mrcg = self.transform(mrcg)
        #     soundFormatted = mrcg.float()
        #     # soundFormatted = torch.squeeze(soundFormatted)
        #     # soundFormatted.transpose_(0, 1)  
        #     print("mrcg shape", soundFormatted.shape)     
        #     return soundFormatted, self.labels[index]

        soundFormatted = torch.from_numpy(mrcg).float()

        # soundFormatted.transpose_(0, 1)
        soundFormatted = soundFormatted.unsqueeze(dim=0)
        soundFormatted = torch.cat((soundFormatted,soundFormatted,soundFormatted))
        # print("mrgc shape", soundFormatted.shape)     
        return soundFormatted, self.labels[index]
  
    def __len__(self):
        return len(self.file_names)
