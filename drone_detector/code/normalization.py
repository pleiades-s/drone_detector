import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio

import dronedataset

# %matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) 

name = ['1.0', '0.9', '0.8', '0.7', '0.6'] #,'0.5','0.4','0.3','0.2','0.1'

SR = 44100
batch_size = 64


for path in name:

    csv_path = '/home/stealthdrone/Desktop/data/csv/' + path + '.csv' 
    file_path = '/home/stealthdrone/Desktop/data/trimmed/' + path 

    metaData = pd.read_csv(csv_path)
    dataset = dronedataset.MFCC_1D(csv_path, file_path, range(1, 7), range(0,2))

    print("Dataset size: " + str(len(dataset)))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, **kwargs)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    
    print(mean.shape)
    print(std.shape)

    f = open("/home/stealthdrone/Desktop/output_log/MFCC_mean_std_"+path, "w")

    f.write("MEAN\n{}\n\n".format(mean))
    f.write("STD\n{}\n\n".format(std))

    f.close()
