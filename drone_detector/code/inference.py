import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import print_test
import test_dronedataset
import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#Config-------------------#
test_batch = 16
#-------------------------#
data_type = 'MFCC_1D'

# argv check
if(len(sys.argv) != 2):
    print("WRONG ARGUMENT VECTOR")
    exit()

model_dir_path = sys.argv[1] 

#split by '/'
model_path = model_dir_path.split('/')[-1]
print("model_path", model_path)
# model_path = 'Conv_Lstm_mfcc_1_2019-11-19_21_03:47:46.pth'

model_name = model_path.split("_20")[0]
# model_name = 'Conv_Lstm_mfcc_1'

data_length = str(float(model_name.split('mfcc_')[1]) / 10)
# data_length_list = ['0.1']

print("model_dir_path: {}\nmodel_path: {}\nmodel_name: {}\ndata_length: {}"
.format(model_dir_path, model_path, model_name, data_length))




# for data_length in data_length_list:

csv_path = '/home/stealthdrone/Desktop/data/csv/test_' + data_length + '.csv' 
file_path = '/home/stealthdrone/Desktop/data/trimmed/test/' + data_length

    
if model_name != 'resnet50':
    model_class = preprocessing.class_for_name('network', model_name)
    model = model_class()
    print(data_type)
    
else:
    pretrained_model = preprocessing.class_for_name('torchvision.models', model_name)
    model = pretrained_model(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    data_type = 'MFCC_2D'
    print(data_type)
    
dataset_class = preprocessing.class_for_name('test_dronedataset', 'TEST_'+ data_type)
    
model.load_state_dict(torch.load('/home/stealthdrone/Desktop/modeltopi/' + model_path))
model.to(device)
model.eval()

test_set = dataset_class(csv_path, file_path)
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=False, **kwargs)

f = open("/home/stealthdrone/Desktop/output_log/inference_"+ model_name + '_' 
    + data_length + "_output.txt", "w")
print_test.inference_print(model, test_loader, f)

f.close()

print("END")
