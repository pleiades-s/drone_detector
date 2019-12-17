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
data_length_list = ['0.1']
test_batch = 16
model_path = 'Conv_Lstm_mfcc_1_2019-11-21 03:47:46.pth' 
model_name = 'Conv_Lstm_mfcc_1'
#-------------------------#


data_type = ''

for data_length in data_length_list:

    csv_path = '/home/stealthdrone/Desktop/data/csv/test_' + data_length + '.csv' 
    file_path = '/home/stealthdrone/Desktop/data/trimmed/test/' + data_length

    if model_name != 'resnet50':
        model_class = preprocessing.class_for_name('network', model_name)
        model = model_class()
        data_type = 'MFCC_1D'
    
    else:
        pretrained_model = preprocessing.class_for_name('torchvision.models', model_name)
        model = pretrained_model(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 6)
        data_type = 'MFCC_2D'
    
    dataset_class = preprocessing.class_for_name('test_dronedataset', 'TEST_'+ data_type)
    
    model.load_state_dict(torch.load('/home/stealthdrone/Desktop/model/' + model_path))
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

