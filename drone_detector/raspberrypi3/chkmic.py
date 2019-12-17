import pyaudio

po = pyaudio.PyAudio()

for index in range(po.get_device_count()):
    desc=po.get_device_info_by_index(index)
    if "USB Audio" in desc["name"]:
        print ("device: %s index: %s rate:%s" %(desc["name"],index,int(desc["defaultSampleRate"])))
