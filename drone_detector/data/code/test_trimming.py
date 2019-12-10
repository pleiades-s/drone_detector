import numpy as np
import random
import librosa
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import random
import math


SR = 44100
DR_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #second
name_list =['test']

def load_audio_file(file_path):
    data = librosa.core.load(file_path, sr = SR)[0]
    return data

def save_newfile(y, name, Trim_dataDir, DURATION):
    path = Trim_dataDir

    mid = int(len(y) /2 )
    l_data = y[:mid]
    r_data = y[mid:]
    output_length = SR * DURATION
    stride = DURATION * SR
    i = 0
    left_duration = len(y) / 2 - output_length + stride

    while(left_duration >= stride):
        data1 = l_data[int(stride * i) : int(output_length + stride * i)]
        data2 = r_data[int(stride * i) : int(output_length + stride * i)]
        data3 = np.append(data1, data2)

        librosa.output.write_wav(path + "/" + name + "_" + str(i).zfill(3) + ".wav", data3, SR)
        left_duration = left_duration - stride
        i = i + 1
    
for sec in DR_list: # DR_list = [0.1,0.2,0.3,0.4,0.5] 
    for j in name_list: #name_list =['1','2']
        
        # print(path)
        sr_data_dir = '/home/stealthdrone/Desktop/data/augmentation/Drone_audio_' + j

        target_dir = '/home/stealthdrone/Desktop/data/trimmed/test/' + str(sec) + '/'

        

        fileList = np.sort(np.array([x.split(".wav") for x in os.listdir(sr_data_dir)])[:,0])

        fileDic = dict([(fileList[x],x) for x in range(len(fileList))])

        for i in os.listdir(sr_data_dir):
            try:
                name = i.split(".wav")[0]
                location = fileDic[name]
                ##1. load
                y = load_audio_file(sr_data_dir + "/" + i)

                save_newfile(y, name, target_dir ,sec)

            except:
                continue
print('complete')
