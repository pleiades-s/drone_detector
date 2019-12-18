import sys
import numpy as np
import librosa
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import datetime
import time
from random import *


import preprocessing
import dronedataset
import network
import print_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("")
#Config----------------------------------------#
data_type = 'MFCC_1D'
num_epoch = 100
train_batch = 64
val_batch = 16
test_batch = 16
lr = 0.01 #0.1, 0.005
step_size = 30
SR = 44100
#-----------------------------------------------#

x = ['0.1', '0.5', '1.0']
data_length_list = [] #, '0.5', '1.0'
model_list = [] # 'Conv1d_mfcc_5', 'Conv1d_mfcc_10' #모델이랑 data 순서 잘 맞추기

# argv check
for i in range(len(sys.argv)-1):
    if sys.argv[i+1] in x:
        data_length_list.append(sys.argv[i+1])
        model_list.append('Conv1d_mfcc_' + str(int(float(sys.argv[i+1])*10)))
    else:
        print("WRONG ARGUMENT VECTOR")
        exit()



def train(model, epoch):
    model.train()
    avg_loss = 0.

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        # data.transpose_(1, 2)
        data = torch.as_tensor(data, dtype=torch.double, device=device)
        target = torch.as_tensor(target, dtype=torch.long, device=device)
        data = data.requires_grad_() #set requires_grad to True for training
        # print("Data", data.shape)
        output = model(data)
        loss = F.nll_loss(output, target)
        avg_loss += loss

        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0: #print training stats
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss))
            f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
        
    print("Average loss: {:.6f}\n".format(avg_loss/len(train_loader)))
    f.write("Average loss: {:.6f}\n".format(avg_loss/len(train_loader)))
    

def validation(model, epoch):
    model.eval()
    correct = 0
    
    for data, target in val_loader:
        # data.transpose_(1, 2)
        model = model.double()
        data = torch.as_tensor(data, dtype=torch.double, device=device)
        target = torch.as_tensor(target, dtype=torch.long, device=device)
        output = model(data)
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    
    acc = 100. * correct / len(val_loader.dataset)

    print('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(val_loader.dataset), acc))
    
    f.write('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(val_loader.dataset), acc))

for idx, data_length in enumerate(data_length_list):
        
    cur_time = str(datetime.datetime.now()).split(".")[0]
    f = open(os.getcwd().split('code')[0]+"output_log/"+ model_list[idx] + '_' 
            + data_length + '_' + cur_time + "_output.txt", "w")

    print("model = {}\ndata_length = {}\nnum_epoch = {}\ntrain_batch = {}\nlr = {}\nstep_size = {}"
        .format(model_list[idx], data_length,num_epoch,train_batch, lr, step_size))
    f.write("model = {}\ndata_length = {}\nnum_epoch = {}\ntrain_batch = {}\nlr = {}\nstep_size = {}"
        .format(model_list[idx], data_length,num_epoch,train_batch, lr, step_size))
    #PATH-------------------------------------------#
    csv_path = os.getcwd().split('code')[0]+'data/csv/' + data_length + '.csv' 
    file_path = os.getcwd().split('code')[0]+'data/trimmed/' + data_length
    #-----------------------------------------------#

    #LOAD DATA-----------------------------------------#
    # mean, std = preprocessing.mean_std_tensor(1, '/home/stealthdrone/Desktop/output_log/MFCC_mean_std_' + data_length)

    # normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # test_normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(test_mean, test_std)])

    preprocessing.fold_shuffle(csv_path)

    dataset_class = preprocessing.class_for_name('dronedataset', data_type)
    rand_num = randint(1, 6)

    train_set = dataset_class(csv_path, file_path, range(1,7), [0])
    val_set = dataset_class(csv_path, file_path, range(1,7), [1])
    test_set = dataset_class(csv_path, file_path, [rand_num], [1])

    print("Train set size: " + str(len(train_set)))
    print("Validation set size: " + str(len(val_set)))
    print("Test set size: " + str(len(test_set)))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = train_batch, shuffle = True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = val_batch, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_batch, shuffle = True, **kwargs)
    #-----------------------------------------------#

    #NEURAL NETWORK---------------------------------#
    model = preprocessing.class_for_name('network', model_list[idx])()
    model.to(device)
    model = model.double()
    print(model)
    f.write(str(model))

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = 0.1)
    #-----------------------------------------------#

    #-----------------------------------------------#
    since = time.time()

    f.write("Start: {}\n".format(str(datetime.datetime.now()).split(".")[0]))
    print("Start: {}".format(str(datetime.datetime.now()).split(".")[0]))

    f.write(str(model))

    log_interval = 20

    #TRAIN AND VALIDATION---------------------------#
    for epoch in range(1, num_epoch + 1):
        
        train(model, epoch)
        scheduler.step()
        validation(model, epoch)

    time_elapsed = time.time() - since

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    f.write('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #TEST--------------------------------------------#
    avg_acc = print_test.test_acc(model,  test_loader)
    f.write('Accuracy of the network on the testset: {:.6f}%\n'.format(avg_acc))
    outcome = print_test.test_acc_classes(model, test_loader, len(test_set), test_batch, f)
    print_test.print_confusion_matrix(outcome, f)
    #-----------------------------------------------#

    f.write("End: {}\n".format(str(datetime.datetime.now()).split(".")[0]))
    print("End: {}".format(str(datetime.datetime.now()).split(".")[0]))

    f.close()

    torch.save(model.state_dict(), os.getcwd().split('code')[0]+'model/'+model_list[idx]+'_' + cur_time + '.pth')
    print("Model is successfully saved.")
