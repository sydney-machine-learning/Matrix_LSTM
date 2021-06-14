import matplotlib.pyplot as plt
import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from MNN_from_scratch import MNN, Network
from fastai.basics import *
import sys

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled,y = shuffle(X_scaled,y,random_state=5)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle = False, stratify = None)
X_train, X_test, Y_train, Y_test =map(torch.LongTensor, (X_train, X_test, Y_train, Y_test))

class data_set(Dataset):                                         #Custom dataset class for dataloader
    def __init__(self,trainx,labelx):
        data=[]
        for i in range(len(trainx)):
            data.append([trainx[i],labelx[i]])
        self.data = data
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]
    
trainset = data_set(X_train,Y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False)          

valset = data_set(X_test,Y_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False)


input_dims = [1,4]
hidden_dims = [16,16]
hidden_dims2 = [8, 8]
output_dims = [1,3]

model = Network(input_dims, hidden_dims,hidden_dims2, output_dims)
lr=2e-2
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


def update(x,y,lr):
    
    y_hat = model(x.float())
    y_hat= torch.squeeze(y_hat,dim=1)   #y_hat has 1 extra dimension
    
    loss = loss_func(y_hat, y) 
    loss.requres_grad = True
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(),y_hat

train_losses = []
train_accuracy = []
val_accuracy = []
epochs = 2000
for j in range(epochs): 
 
    los=0;
    correct_train=0
    total_train=0
    correct_val=0
    total_val=0
    for data, labels in trainloader:

        data = data.unsqueeze(1)                #Needs 1 extra dimension to make data column 3 dimensional (4x1x4)
        los_t,output = update(data,labels,lr)
        los=los+los_t

        for k in range(output.size()[0]):
            ma = 0
            for l in range(3):
                if output[k][l]>output[k][ma]:
                    ma=l
            if labels[k] == ma:
                correct_train=correct_train+1  
            total_train+=1
    for data,labels in valloader:
        data = data.unsqueeze(1)
        with torch.no_grad():
            output = model(data.float())

            output=output.squeeze(1)
            for k in range(output.size()[0]):
                ma = 0
                for l in range(3):
                    if output[k][l]>output[k][ma]:
                        ma=l
                if labels[k] == ma:
                    correct_val+=1  
                total_val+=1
    print(f'epoch ====> {j}')
    train_losses.append(los)                                      #stores training loss
    train_accuracy.append((correct_train/total_train)*100)
    val_accuracy.append((correct_val/total_val)*100)  



           

plt.plot(train_accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label = 'Validation Accuracy')
plt.legend()
plt.savefig(r'D:\python\img\mnn_iris_accuracy.jpg')        #Edit graph location 
