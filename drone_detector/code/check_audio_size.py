import librosa
import preprocessing
import MRCG

input_size =[1.0]
input_data = []
SR = 44100

path = '/home/stealthdrone/Desktop/data/rawdata/1-L.wav' 
data = librosa.core.load(path, sr = SR, mono = True)[0]
#pre-emphasis
data = preprocessing.preemphasis(data)

for i, length in enumerate(input_size):
    input_data.append(data[:int(length * SR * 2)])
    

mfcc = librosa.feature.mfcc(y = data, sr = SR, hop_length = 1024,
                             n_mfcc = 40)

for i, data in enumerate(input_data):
    print("Audio Length: ", librosa.get_duration(y=data, sr=SR))

    #stft
    stft = librosa.core.stft(y=data, hop_length = 1024)
    print("STFT", stft.shape)
    #mfcc
    mfcc = librosa.feature.mfcc(y = data, sr = SR, hop_length = 1024, n_mfcc = 40)
    print("MFCC", mfcc.shape)

    #mrcg
    mrcg=MRCG.mrcg_extract(data, SR)
    print("MRCG", mrcg.shape)

    print("\n\n")
