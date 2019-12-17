import csv
import os

dominant = os.getcwd() + '/'
name_list = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']

for name in name_list:
    f=open(dominant + name +'.csv','a',  encoding='utf-8',newline='')
    wr=csv.writer(f)
    wr.writerow(['filename','class','fold'])
    f.close()
