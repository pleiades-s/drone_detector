import pyaudio
import wave
import multiprocessing
import sys
import time
from multiprocessing import Process

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
#RECORD_SECONDS = 1000
RECORD_SECONDS = 10
def record(INDEX,TIME,NAME):
    po = pyaudio.PyAudio()
    ###
    #print(INDEX,' befpre po.opne')
    #lock for the same time
    #a=time.time()
    #while a <= TIME:
    #    a=time.time()
    #print(INDEX,a)
    
    TIME=TIME+1
    
    stream = po.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=INDEX,frames_per_buffer=CHUNK)
    frames = []
    

    #lock for the same time
    a=time.time()
    while a <= TIME:
        a=time.time()
    print('before stream.read',INDEX,a)
    TIME=TIME+1

    for i in range(0,int(RATE/CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)


    stream.stop_stream()
    stream.close()
    po.terminate()

    wavefile=wave.open(NAME+'.wav','wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(po.get_sample_size(FORMAT))
    wavefile.setframerate(RATE)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

if __name__=='__main__':
    
    TIME = time.time() + 2
    p1 = Process(target = record, args = (2,TIME,sys.argv[1]))
    p2 = Process(target = record, args = (3,TIME,sys.argv[2]))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

