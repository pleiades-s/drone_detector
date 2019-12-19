import os


#namelist=['Trimmed_audio_1','Trimmed_audio_2','Trimmed_audio_ps1_1','Trimmed_audio_ps1_2','Trimmed_audio_ps2_1','Trimmed_audio_ps2_2','Trimmed_audio_ts_1','Trimmed_audio_ts_2']


#namelist =['Drone_audio_' ,'Drone_audio_ts_' ,'Drone_audio_ps1_' ,'Drone_audio_ps2_' ]

namelist = ['Drone_audio_', 'Drone_audio_aug_']
#Drlist=['0.4','0.3','0.2','0.1']
Drlist=['1','2','3','4','5','6']

for name in namelist:
    for dr in Drlist:
        try:
            os.makedirs(os.path.join(name+dr))
        except OSError as e:
            print("fail")

try:
    os.makedirs(os.path.join('Drone_audio_test'))
except OSError as e:
    print("fail")
