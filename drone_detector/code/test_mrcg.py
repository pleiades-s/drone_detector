import numpy as np
import MRCG
import scipy.io.wavfile
import os
import librosa
import wave
import time
SR = 44100

Test_Audio_Path = '/home/stealthdrone/Desktop/dataset/Drone_audio_1/1-1.wav'

test_audo, sr = librosa.load(Test_Audio_Path, sr = SR)

output=MRCG.mrcg_extract(test_audo,SR)

print(output.shape)
print(output.size)

