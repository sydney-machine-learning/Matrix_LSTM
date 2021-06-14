
import matplotlib.pyplot as plt
import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from MNN_from_scratch import MNN, Network
from fastai.basics import *
import sys

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
PATH_TO_STORE_TRAINSET = r'D:\python\FMNIST\train'            #Edit dataset folder locations
PATH_TO_STORE_TESTSET = r'D:\python\FMNIST\test'
trainset = datasets.FashionMNIST(PATH_TO_STORE_TRAINSET, download=False, train=True, transform=transform)       #Set download True
valset = datasets.FashionMNIST(PATH_TO_STORE_TESTSET, download=False, train=False, transform=transform)         #If dataset not downloaded 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)





input_dims = [28,28]
hidden_dims = [32,32]
hidden_dims2 = [16, 16]
output_dims = [1,10]

model = Network(input_dims, hidden_dims,hidden_dims2, output_dims)

lr=2e-2
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)



def update(x,y,lr):
    
    y_hat = model(x)
    y_hat= torch.squeeze(y_hat,dim=1)  #y_hat has 1 extra dimension
   
    loss = loss_func(y_hat, y) 
    loss.requres_grad = True
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(),y_hat

train_losses = []
train_accuracy = []
val_accuracy = []
epochs =300

for j in range(epochs): 
    los=0;
    correct_train=0
    total_train=0
    correct_val=0
    total_val=0
    for images, labels in trainloader:
        los_t,output = update(images[:,0],labels,lr)          #images dimension (64x1x28x28)
        los=los+los_t                                         # ignore 2nd dimension (1)

        for k in range(output.size()[0]):
            ma = 0

            for l in range(10):
                if output[k][l]>output[k][ma]:
                    ma=l
            if labels[k] == ma:
                correct_train=correct_train+1   
            total_train+=1
    for images,labels in valloader:
        with torch.no_grad():
            output = model(images.float()[:,0])

            output=output.squeeze(1)
      
            for k in range(output.size()[0]):
                ma = 0

                for l in range(10):
                    if output[k][l]>output[k][ma]:
                        ma=l
                if labels[k] == ma:
                    correct_val+=1  
                total_val+=1
    print(f'epoch ====> {j}')
    train_losses.append(los)
    train_accuracy.append((correct_train/total_train)*100)
    val_accuracy.append((correct_val/total_val)*100)  



           


plt.plot(train_accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label = 'Validation Accuracy')
plt.legend()
plt.savefig(r'D:\python\img\mnn_FashionMNIST_accuracy.jpg')              #Edit graph location
