import copy
import time
import librosa
import datetime
from tqdm import tqdm
from random import *

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import preprocessing
import dronedataset
import network
import print_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("")
#Config----------------------------------------#
data_type = 'MFCC_2D'
pre_trained = 'resnet50'
num_epoch = 60
train_batch = 64
val_batch = 16
test_batch = 16
lr = 0.01
step_size = 30
#-----------------------------------------------#
x = ['0.1', '0.5', '1.0']
data_length_list = [] 


# argv check
for i in range(len(sys.argv)-1):
    if sys.argv[i+1] in x:
        data_length_list.append(sys.argv[i+1])
    else:
        print("WRONG ARGUMENT VECTOR")
        exit()

def train_model(model, criterion, optimizer, scheduler, data_length, num_epochs=num_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        f.write('Epoch {}/{}\n'.format(epoch + 1, num_epochs))
        f.write('-' * 10)
        f.write('\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = torch.as_tensor(inputs, dtype=torch.float, device=device)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # del outputs
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # print("Working")

                    # del outputs

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))    

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    

    f.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    f.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 
                os.getcwd().split('code')[0]+'model/'+ pre_trained + '_' + data_length + '_' 
                +cur_time + '.pth')
    print("Transfer learning model is successfully saved.\n\n")

    return model


for data_length in data_length_list:

    cur_time = str(datetime.datetime.now()).split(".")[0]
    f = open(os.getcwd().split('code')[0]+"output_log/" + pre_trained + '_' + data_length 
            + '_' + cur_time + "_output.txt", "w")

    print("data_length = {}\nnum_epoch = {}\ntrain_batch = {}\nlr = {}\nstep_size = {}\n"
        .format(data_length,num_epoch,train_batch, lr, step_size))
    f.write("data_length = {}\nnum_epoch = {}\ntrain_batch = {}\nlr = {}\nstep_size = {}\n"
        .format(data_length,num_epoch,train_batch, lr, step_size))

    #PATH-------------------------------------------#
    csv_path = os.getcwd().split('code')[0]+'data/csv/' + data_length + '.csv' 
    file_path = os.getcwd().split('code')[0]+'data/trimmed/' + data_length
    #-----------------------------------------------#

    #LOAD DATA-----------------------------------------#
    
    #WHEN ONLY MFCC
    #mean, std = preprocessing.mean_std_tensor(1, os.getcwd().split('code')[0]+'output_log/MFCC_mean_std_' + data_length)

    #normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    preprocessing.fold_shuffle(csv_path)
    rand_num = randint(1, 6)
    dataset_class = preprocessing.class_for_name('dronedataset', data_type)

    train_set = dataset_class(csv_path, file_path, range(1,7), [0],transform=False)
    val_set = dataset_class(csv_path, file_path, range(1,7), [1], transform=False)
    test_set = dataset_class(csv_path, file_path, [rand_num], [1], transform=False)

    print("Train set size: " + str(len(train_set)))
    print("Validation set size: " + str(len(val_set)))
    print("Test set {} size: {}".format(rand_num, str(len(test_set))))
    
    f.write("Train set size: \n" + str(len(train_set)))
    f.write("Validation set size: \n" + str(len(val_set)))
    f.write("Test set {} size: {}\n".format(rand_num, str(len(test_set))))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = train_batch, shuffle = True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = val_batch, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_batch, shuffle = True, **kwargs)


    image_datasets = {'train':train_set, 'val':val_set}

    dataloaders = {'train':train_loader, 'val':val_loader}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ["section 1 forward", "section 1 hovering",
                    "section 2 forward", "section 2 hovering",
                    "section 3 forward", "section 3 hovering"]

    pretrained_model = preprocessing.class_for_name('torchvision.models', pre_trained)
    model_ft = pretrained_model(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    #criterion = F.nll_loss

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay = 0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data_length, num_epochs=num_epoch)

    #TEST--------------------------------------------#
    avg_acc = print_test.test_acc(model_ft,  test_loader)
    f.write('Accuracy of the network on the testset: {:.6f}%\n'.format(avg_acc))
    outcome = print_test.test_acc_classes(model_ft, test_loader, len(test_set), test_batch, f)
    print_test.print_confusion_matrix(outcome, f)
    #-----------------------------------------------#
    print("END\n\n")
    f.close()
