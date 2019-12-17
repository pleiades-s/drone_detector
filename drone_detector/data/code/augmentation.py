import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import csv
import pandas as pd
import os

dominant = []
num_experiment = 6

for i in range(num_experiment):
    dominant.append('Drone_audio_' + str(i+1))

SR = 44100

def load_audio_file(file_path):
    data = librosa.core.load(file_path, sr = SR)[0] #, sr=44100
    print(file_path)
    return data

def augmentation(y):
    data = []
    #time
    data.append(librosa.effects.time_stretch(y, 0.81))
    data.append(librosa.effects.time_stretch(y, 0.93))
    data.append(librosa.effects.time_stretch(y, 1.07))
    data.append(librosa.effects.time_stretch(y, 1.23))
    #ps1
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=-2))
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=-1))
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=1))
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=2))
    #ps2
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=-3.5))
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=-2.5))
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=2.5))
    data.append(librosa.effects.pitch_shift(y, SR, n_steps=3.5))

    return data

def save_newfile(y, name, label, fold, csv_path):

    path = os.getcwd().split('code')[0]+'augmentation/Drone_audio_aug_' + fold

    print(path)
    nickname = ['ts_1', 'ts_2', 'ts_3','ts_4',
                   'ps1_1', 'ps1_2', 'ps1_3', 'ps1_4',
                   'ps2_1', 'ps2_2', 'ps2_3', 'ps2_4']

    f = open(csv_path, 'a', encoding='utf-8', newline='')
    # print("write")

    wr = csv.writer(f)
    # print("after write")

    for i in range(12):
        filename = nickname[i] + '_' + name
        librosa.output.write_wav(path + "/" + filename + ".wav", y[i], SR)
        print(path + "/" + filename + ".wav")
        wr.writerow([filename, label, fold])

    f.close()


for n, path in enumerate(dominant):

    dataDir = os.getcwd().split('code')[0]+'augmentation/' + path
    csvPath = os.getcwd().split('code')[0]+'csv/' + path + '.csv'

    #data dir iteration
    metadata = pd.read_csv(csvPath, sep=",")
    metadata = metadata.sort_values(by=['filename'])
    metadata = metadata.reset_index(drop=True)

    fileList = np.sort(np.array([x.split(".wav") for x in os.listdir(dataDir)])[:,0])

    fileDic = dict([(fileList[x],x) for x in range(len(fileList))])

    for i in os.listdir(dataDir):
        try:
            name = i.split(".wav")[0]
            
            location = fileDic[name]

            print('##1')
            ##1. load audio
            y = load_audio_file(dataDir + "/" + i)
            print('##2')
            ##2. augment audio
            data = augmentation(y)


            print("##3")
            ##3. save new audio  
            sound_type = metadata.loc[location]["class"]
            print(sound_type)
            save_newfile(data, name, sound_type, str(n+1), csvPath) # fold: range(1, 9) :1 ~ 8
            
        except:
            print("Error")

print("Completed.")

