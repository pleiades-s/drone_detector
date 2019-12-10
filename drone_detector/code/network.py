# Network structures

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d_mfcc_5(nn.Module):

    def __init__(self):
        super(Conv1d_mfcc_5, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 44 x 40 -> 44 x 64
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 44 x 64 -> 22 x 64
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 22 x 64 -> 22 x 128
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 22 x 128 -> 11 x 128
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # 11 x 128 -> 11 x 256
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)                    # 11 x 256 -> 5 x 256
        self.conv4 = nn.Conv1d(256, 512, 3, padding=1)  # 5 x 256 -> 5 x 512
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(5)                    # 5 x 512 -> 1 x 512
        self.fc1 = nn.Linear(512, 64)                   
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 6)                     # 1 x 64 -> 1 x 6
        
    def forward(self, x):
        #print(x.size())      
        x = self.conv1(x)
        #print(x.size())
        x = F.relu(self.bn1(x))
        #print(x.size())
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#


        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = F.relu(self.bn2(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv3(x)
        #print(x.size())
        x = F.relu(self.bn3(x))
        #print(x.size())
        x = self.pool3(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv4(x)
        #print(x.size())
        x = F.relu(self.bn4(x))
        #print(x.size())
        x = self.pool4(x)
        # x = self.dropout1(x)
        #----------------------------#
        
        # x = self.avgPool(x)
        #print(x.size())
        x = x.view(-1, 512 * 1)
        # print("x shape", x.shape)
        #print(x.size()) #change the 512x1 to 1x512
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        #----------------------------#

        #print(x.size())
        x = self.fc2(x)
        #----------------------------#

        #print(x.size())
        x = F.log_softmax(x, dim=1)
        #print(x.size())
        return x

class Conv1d_mfcc_1(nn.Module):#input size [40, 9]

    def __init__(self):
        super(Conv1d_mfcc_1, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 9 x 40 -> 9 x 64
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 9 x 64 -> 4 x 64
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 4 x 64 -> 4 x 128
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 4 x 128 -> 2 x 128
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # 2 x 128 -> 2 x 256
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3= nn.AvgPool1d(2)                     # 2 x 256 -> 1 x 256
        self.fc1 = nn.Linear(256, 64)                   
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 6)                     # 1 x 64 -> 1 x 6
        
    def forward(self, x):
        #print(x.size())      
        x = self.conv1(x)
        #print(x.size())
        x = F.relu(self.bn1(x))
        #print(x.size())
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#


        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = F.relu(self.bn2(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv3(x)
        #print(x.size())
        x = F.relu(self.bn3(x))
        #print(x.size())
        x = self.pool3(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#
        #----------------------------#
        
        # x = self.avgPool(x)
        #print(x.size())
        x = x.view(-1, 256 * 1)
        # print("x shape", x.shape)
        #print(x.size()) #change the 512x1 to 1x512
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        #----------------------------#

        #print(x.size())
        x = self.fc2(x)
        #----------------------------#

        #print(x.size())
        x = F.log_softmax(x, dim=1)
        #print(x.size())
        return x

class Conv1d_mfcc_10(nn.Module):#input size [40, 87]

    def __init__(self):
        super(Conv1d_mfcc_10, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 87 x 40 -> 87 x 64
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 87 x 64 -> 43 x 64
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 43 x 64 -> 43 x 128
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 43 x 128 -> 21 x 128
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # 21 x 128 -> 21 x 256
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)                    # 21 x 256 -> 10 x 256
        self.conv4 = nn.Conv1d(256, 512, 3, padding=1)  # 10 x 256 -> 10 x 512
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(2)                    # 10 x 512 -> 5 x 512
        self.conv5 = nn.Conv1d(512, 1024, 3, padding=1) # 5 x 512 -> 5 x 1024
        self.bn5 = nn.BatchNorm1d(1024)
        self.pool5 = nn.AvgPool1d(5)                    # 5 x 1024 -> 1 x 1024
        self.fc1 = nn.Linear(1024, 64)                   
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 6)                     # 1 x 64 -> 1 x 6
        
    def forward(self, x):
        #print(x.size())      
        x = self.conv1(x)
        #print(x.size())
        x = F.relu(self.bn1(x))
        #print(x.size())
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#


        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = F.relu(self.bn2(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv3(x)
        #print(x.size())
        x = F.relu(self.bn3(x))
        #print(x.size())
        x = self.pool3(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv4(x)
        #print(x.size())
        x = F.relu(self.bn4(x))
        #print(x.size())
        x = self.pool4(x)
        # x = self.dropout1(x)
        #----------------------------#
        #print(x.size())      
        x = self.conv5(x)
        #print(x.size())
        x = F.relu(self.bn5(x))
        #print(x.size())
        x = self.pool5(x)
        # x = self.dropout1(x)
        #----------------------------#

        
        # x = self.avgPool(x)
        #print(x.size())
        x = x.view(-1, 1024 * 1)
        # print("x shape", x.shape)
        #print(x.size()) #change the 512x1 to 1x512
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        #----------------------------#

        #print(x.size())
        x = self.fc2(x)
        #----------------------------#

        #print(x.size())
        x = F.log_softmax(x, dim=1)
        #print(x.size())
        return x


class Conv1d_mfcc_4(nn.Module): #input size [40, 35] 
    def __init__(self):
        super(Conv1d_mfcc_5, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 40 x 35 -> 64 x 35
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 64 x 35 -> 64 x 17
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 64 x 17 -> 128 x 17
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 128 x 17 -> 128 x 8
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # 128 x 8 -> 256 x 8
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)                    # 256 x 8 -> 256 x 4
        self.conv4 = nn.Conv1d(256, 512, 3, padding=1)  # 256 x 4 -> 512 x 4
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(4)                    # 512 x 4 -> 512 x 1
        self.fc1 = nn.Linear(512, 64)                   # 512 -> 64
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 6)                     # 64 -> 6
        
    def forward(self, x):
        #print(x.size())      
        x = self.conv1(x)
        #print(x.size())
        x = F.relu(self.bn1(x))
        #print(x.size())
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#


        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = F.relu(self.bn2(x))
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv3(x)
        #print(x.size())
        x = F.relu(self.bn3(x))
        #print(x.size())
        x = self.pool3(x)
        #print(x.size())
        # x = self.dropout1(x)
        #----------------------------#


        x = self.conv4(x)
        #print(x.size())
        x = F.relu(self.bn4(x))
        #print(x.size())
        x = self.pool4(x)
        # x = self.dropout1(x)
        #----------------------------#
        
        # x = self.avgPool(x)
        #print(x.size())
        x = x.view(-1, 512 * 1) #???????????????????? input size 확인하기
        print("x shape", x.shape) # [batch, 512] ????????????????????
        #print(x.size()) #change the 512x1 to 1x512
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        #----------------------------#

        #print(x.size())
        x = self.fc2(x)
        #----------------------------#

        #print(x.size())
        x = F.log_softmax(x, dim=1)
        #print(x.size())
        return x

# class Conv1d_mfcc_3(nn.Module): #input size [40, 26]
# class Conv1d_mfcc_2(nn.Module): #input size [40, 18]


class Conv_Lstm_mfcc_5(nn.Module):
    def __init__(self):
        super(Conv_Lstm_mfcc_5, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 44 x 40 -> 44 x 64
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 44 x 64 -> 22 x 64
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 22 x 64 -> 22 x 128
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 22 x 128 -> 11 x 128
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # 11 x 128 -> 11 x 256
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)                    # 11 x 256 ->  5 x 256
        
        self.lstm = nn.LSTM(256, 64, num_layers=2, bidirectional=False, dropout=0.1)
        self.hidden2out = nn.Linear(5 * 64, 64)
        self.dropout5 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(64, 6)
        self.softmax = nn.LogSoftmax()


    def forward(self, x):
              
        x = self.conv1(x)
        # print("size", x.shape)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        # x = self.dropout2(x)
        # print("size", x.shape)
        #----------------------------#
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        Batch, Freq, Time = x.size()
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = x.reshape(Batch, -1)
        x = F.relu(self.hidden2out(x))
        x = self.dropout5(x)
        x = self.fc1(x)
        
        # print("output size: ", x.size())
        x = F.log_softmax(x, dim=1)
        #----------------------------#
        return x

class Conv_Lstm_mfcc_1(nn.Module): # [40, 9]

    def __init__(self):
        super(Conv_Lstm_mfcc_1, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 9 x 40 -> 9 x 64
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 9 x 64 -> 4 x 64
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 4 x 64 -> 4 x 128
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 4 x 128 -> 2 x 128
        #--------------------------------------------------------------------------------------------------#
        self.lstm = nn.LSTM(128, 64, num_layers=2, bidirectional=False, dropout=0.1)
        # self.dropout4 = nn.Dropout(p=0.1)
        self.hidden2out = nn.Linear(64 * 2, 64)
        self.dropout5 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(64, 6)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
              
        x = self.conv1(x)
        # print("size", x.shape)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        #----------------------------#
        Batch, Freq, Time = x.size()
        x = x.permute(0, 2, 1)
        # print("output size: ", x.size())
        x, _ = self.lstm(x)

        x = x.reshape(Batch, -1)
        x = F.relu(self.hidden2out(x))
        x = self.dropout5(x)
        x = self.fc1(x)
        
        # print("output size: ", x.size())
        x = F.log_softmax(x, dim=1)
        #----------------------------#
        return x

class Conv_Lstm_mfcc_10(nn.Module): # [40, 87]

    def __init__(self):
        super(Conv_Lstm_mfcc_10, self).__init__()
        self.conv1 = nn.Conv1d(40, 64, 3, padding=1)    # 87 x 40 -> 87 x 64
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)                    # 87 x 64 -> 43 x 64
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)   # 43 x 64 -> 43 x 128
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)                    # 43 x 128 -> 21 x 128
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  # 21 x 128 -> 21 x 256
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)                    # 21 x 256 -> 10 x 256
        #--------------------------------------------------------------------------------------------------#
        self.lstm = nn.LSTM(256, 64, num_layers=2, bidirectional=False, dropout=0.1)
        # self.dropout4 = nn.Dropout(p=0.1)
        self.hidden2out = nn.Linear(10 * 64, 64)
        self.dropout5 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(64, 6)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
              
        x = self.conv1(x)
        # print("size", x.shape)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        # x = self.dropout2(x)
        # print("size", x.shape)
        #----------------------------#
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        # x = self.dropout3(x)
        #----------------------------#
        Batch, Freq, Time = x.size()
        x = x.permute(0, 2, 1)
        # print("output size: ", x.size())
        x, _ = self.lstm(x)

        x = x.reshape(Batch, -1)
        x = F.relu(self.hidden2out(x))
        x = self.dropout5(x)
        x = self.fc1(x)
        
        # print("output size: ", x.size())
        x = F.log_softmax(x, dim=1)
        #----------------------------#
        return x

# class Conv_Lstm_mfcc_4(nn.Module):
# class Conv_Lstm_mfcc_3(nn.Module):
# class Conv_Lstm_mfcc_2(nn.Module):

class Conv2d_mfcc_5(nn.Module):
    def __init__(self):
        #, output_size, embedding_dim, hidden_dim, num_layers=1
        super(Conv2d_mfcc_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1))    # 1 x 44 x 40 -> 32 x 44 x 40
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=(1,1))   # 32 x 44 x 40 -> 32 x 22 x 20
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1))   # 32 x 22 x 20 -> 64 x 22 x 20
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 64 x 22 x 20 -> 64 x 11 x 10
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))   # 64 x 11 x 10 -> 64 x 11 x 10
        self.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 64 x 11 x 10 -> 64 x 5 x 5
        self.dropout3 = nn.Dropout(p=0.1)
        #--------------------------------------------------------------------------------------------------#
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.dropout4 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(64, 6)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
              
        x = self.conv1(x)
        # print("size", x.shape)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        # x = self.dropout1(x)
        #----------------------------#
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        # x = self.dropout2(x)
        # print("size", x.shape)
        #----------------------------#
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        # x = self.dropout3(x)
        #----------------------------#
        # print("size", x.shape)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        #----------------------------#
        return x