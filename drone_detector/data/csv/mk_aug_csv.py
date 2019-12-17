import csv
import os

dominant = os.getcwd() + '/Drone_audio_'
name_list = [1,2,3,4,5,6]

for name in name_list:
    f=open(dominant + str(name) +'.csv','a',  encoding='utf-8',newline='')
    wr=csv.writer(f)
    label = 0
    wr.writerow(['filename','class','fold'])
    if name != 2:
        for i in range(1,4):
            for j in range(1,5):
                if j == 4 :
                    label+=1
                    final= str(name)+'-'+str(i)+'-h'
                    wr.writerow([final,label,name])
                    label+=1
                else : 
                    final= str(name)+'-'+str(i)+'-'+str(j)
                    wr.writerow([final,label,name])
    else :
        for i in range (1,4):
            final = str(name)+'-'+str(i)+'-2'
            wr.writerow([final,label,name])
            label+=1
            final = str(name)+'-'+str(i)+'-h'
            wr.writerow([final,label,name])
            label+=1
    f.close()
