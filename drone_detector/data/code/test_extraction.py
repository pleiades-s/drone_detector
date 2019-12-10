import librosa
import numpy as np
from test_timetable import start, end
#from timetable import start, end

WORKING_DIR = '/home/stealthdrone/Desktop/data/'
ROW_AUDIO_PATH=WORKING_DIR + 'rawdata/'
SR = 44100
AUG=WORKING_DIR + 'augmentation/Drone_audio_'
TARGET_PATH_LIST = ['','test/']

idx_list = [1,3,4,6]
k=1


def min2sec(time): # change minute to seconds
    minute = int(time.split(':')[0])
    second = int(time.split(':')[1])
    milisec = int(time.split(':')[2])

    return minute * 60 + second + milisec / 100


for j in idx_list :
    left_audio = librosa.core.load(ROW_AUDIO_PATH + str(j) + '-L.wav', sr = SR)[0]
    right_audio = librosa.core.load(ROW_AUDIO_PATH + str(j) + '-R.wav', sr = SR)[0]

    for i in range(1, len(start[j])):
        
        l_start = int(min2sec(start[j][i]) * SR)
        l_end= int(min2sec(end[j][i]) * SR)

        r_start = l_start
        r_end = l_end

        data1 = left_audio[l_start:l_end]
        data2 = right_audio[r_start:r_end]

        data3 = np.append(data1, data2)
        librosa.output.write_wav(AUG + TARGET_PATH_LIST[1] + str(j) + '-' + str(i) + ".wav", data3, SR)
