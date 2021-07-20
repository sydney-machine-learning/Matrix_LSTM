import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from RNN_model import Network
from MovingMNIST_data_processing import trainloader, valloader
from PIL import Image

input_dim = 64*64
hidden_dim = 16*16
output_dim = 64*64

model = Network(input_dim, hidden_dim, output_dim) 
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 200
losses = []
for i in range(epochs):
    for seq, labels in trainloader:
        seq=torch.flatten(seq,-2)
        labels=torch.flatten(labels,-2)

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        losses.append(single_loss.item())
        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()
    print(i)
    if i%5 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        



with torch.no_grad():
    for seq,label in trainloader:
        y_pred = model(torch.flatten(seq,-2))
        
        for i in range(32):
            for j in range(10):
                image_t1 = seq[i,j].numpy()       #For saving the sequence plots
                image_t1*=255
                image_t1=image_t1.astype(int)
                plt.gray()
                plt.imshow(image_t1, interpolation='nearest')
                plt.savefig(r'RNN\images\plots\seq' + f'{i}_{j}.png')    #edit file location here       
                
                
            
            image_t2 = label[i].numpy()           #For saving the label plot corresponding to current sequence
            image_t2*=255
            image_t2=image_t2.astype(int)
            plt.gray()
            plt.imshow(image_t2, interpolation='nearest')
            plt.savefig(r'RNN\images\plots\label' + f'{i}.png')    #edit file location here  
        
            
            image_t3 = np.reshape(y_pred[i].numpy(),(64,64))     #For saving the prediction plot corresponding to current sequence
            image_t3*=255
            image_t3=image_t3.astype(int)
            plt.imshow(image_t3, interpolation='nearest')
            plt.savefig(r'RNN\images\plots\prediction' + f'{i}.png')   #edit file location here  
        
           
            break       #Added breakpoints to plot the predictions for only first sequence. Can be removed for viewing further predictions
            
        break
  
       

#Plots the MSE loss       
plt.plot(losses, label = 'Training MSE losses')
plt.legend()
plt.savefig(r'RNN\images\plots\traininglosses.png')