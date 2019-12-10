x = ['0.1', '0.5', '1.0']
data_length_list = [] #, '0.5', '1.0'
model_list = [] # 'Conv1d_mfcc_5', 'Conv1d_mfcc_10' #모델이랑 data 순서 잘 맞추기

y = ['test.py', '0.1', '0.5']
# argv check
for i in range(len(y)-1):
    if y[i+1] in x:
        data_length_list.append(y[i+1])
        model_list.append('Conv1d_mfcc_' + str(int(float(y[i+1])*10)))
    else:
        print("WRONG ARGUMENT VECTOR")
        exit()

print(data_length_list)
print(model_list)