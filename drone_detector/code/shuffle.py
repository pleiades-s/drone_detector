import math
import random
import sys

def fold_shuffle(path):

    data=[]
    nr_list=[]

    op = open(path,'r')

    for one_line in op.readlines() :
        arr = one_line.split(',',3)
        if(arr[0] == ''):
            continue
        # arr[0] = arr[0][2]
        # arr.append("0\n")
        else: 
            arr[2] = arr[2][0]
            arr.append(' ')
            data.append(arr)

    op.close()

    # print("Data", data)

    # exit()

    data = data[1:]

    for i in range(0, len(data)):
        nr_list.append(i)
    
    sample_ratio = math.floor(len(nr_list) * 0.7)

    s = random.sample(nr_list, sample_ratio)

    # s 에는 random 70 % 들어있음

    for i in range(0, len(nr_list)):
    
        if i in s:

            data[i][3] = '0\n'

        else:
            data[i][3] = '1\n'


    f = open(path,'w')

    f.write("filename,class,fold,tv\n")

    for i in range(0, len(nr_list)):

        f.write(data[i][0]+','+data[i][1]+','+data[i][2]+','+data[i][3])

    f.close()


data_length_list = ['1.0', '0.9', '0.8', '0.7', '0.6','0.5','0.4','0.3','0.2','0.1']

for data_length in data_length_list:
    fold_shuffle('/home/stealthdrone/Desktop/data/csv/' + data_length + '.csv')
