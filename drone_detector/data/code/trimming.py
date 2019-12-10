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
name_list =[1,2,3,4,5,6]

def load_audio_file(file_path):
    data = librosa.core.load(file_path, sr = SR)[0]
    return data

def save_newfile(y, name, label, fold, Trim_dataDir, DURATION, Trim_csv):
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

        librosa.output.write_wav(path + "/" + name + "_" + str(i) + ".wav", data3, SR)
        left_duration = left_duration - stride
        i = i + 1
    
    f = open(Trim_csv, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)

    for k in range(i):
        wr.writerow([name + "_" + str(k), label, fold])
    f.close()




for sec in DR_list: # DR_list = [0.1,0.2,0.3,0.4,0.5] 
    for j in name_list: #name_list =['1','2']
        
        # print(path)
        sr_data_dir = '/home/stealthdrone/Desktop/data/augmentation/Drone_audio_aug_' + str(j)
        sr_csv_path = '/home/stealthdrone/Desktop/data/csv/Drone_audio_'+ str(j) +'.csv'

        target_dir = '/home/stealthdrone/Desktop/data/trimmed/' + str(sec) + '/'
        target_csv = '/home/stealthdrone/Desktop/data/csv/' + str(sec) + '.csv'

        metadata = pd.read_csv(sr_csv_path, sep=",")
        metadata = metadata.sort_values(by=['filename'])
        metadata = metadata.reset_index(drop=True)
        

        fileList = np.sort(np.array([x.split(".wav") for x in os.listdir(sr_data_dir)])[:,0])

        fileDic = dict([(fileList[x],x) for x in range(len(fileList))])

        for i in os.listdir(sr_data_dir):
            try:
                name = i.split(".wav")[0]
                location = fileDic[name]
                ##1. load
                y = load_audio_file(sr_data_dir + "/" + i)
                print(i)
                sound_type = metadata.loc[location]["class"]
                fold = metadata.loc[location]["fold"]

                save_newfile(y, name, sound_type, fold, target_dir ,sec, target_csv)

            except:
                continue
print('complete')
