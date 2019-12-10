# Preprocessing
# 1. get mean, std value from output_log 
# 2. preemphasis

import math
import random
import scipy.signal
import numpy as np
import importlib

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def fold_shuffle(path):

    data=[]
    nr_list=[]

    op = open(path,'r')

    for one_line in op.readlines() :
        arr = one_line.split(',',3)
        arr[2] = arr[2][0]
        arr.append("0\n")
        data.append(arr)

    op.close()

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

def mean_std_tensor(tensor_size, path):
    
    array = []
    mean_tensor = []
    std_tensor = []

    with open(path, "r") as f:

        for line in f:
            array.append([x for x in line.split(', ')])

        count = 0

        for i in range(0, len(array)):
            for j in range(0, len(array[i])):
                ntr = ''.join((ch if ch in '0123456789.-' else '') for ch in array[i][j])
                if count > 0 and count < tensor_size + 1:
                    mean_tensor.append(float(ntr))

                elif count > tensor_size + 2 and count < (tensor_size+2)*2-1:
                    std_tensor.append(float(ntr))

                count = count + 1
        
        f.close()
    
    return mean_tensor, std_tensor

def preemphasis(y, coef=0.97, zi=None, return_zf=False):

    b = np.asarray([1.0, -coef], dtype=y.dtype)
    a = np.asarray([1.0], dtype=y.dtype)

    if zi is None:
        zi = scipy.signal.lfilter_zi(b, a)

    y_out, z_f = scipy.signal.lfilter(b, a, y,
                                      zi=np.asarray(zi, dtype=y.dtype))

    if return_zf:
        return y_out, z_f

    return y_out
