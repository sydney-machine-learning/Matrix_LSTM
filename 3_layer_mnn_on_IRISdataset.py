
import matplotlib.pyplot as plt
import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastai.basics import *
import sys

#load data

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train = X_scaled
Y_train = y

X_train,Y_train =map(torch.LongTensor, (X_train,Y_train))


# standard sigmoid function
def sigmoid(x):
    return 1 / (1+torch.exp(x))







class MNN(nn.Module):
    
    def __init__(self, input_dim_r,input_dim_c,hidden_dim_r, hidden_dim_c,output_dim_r,output_dim_c):
	    
        super().__init__()
        
        self.input_dim_r = input_dim_r
        self.input_dim_c = input_dim_c 
        self.hidden_dim_r= hidden_dim_r
        self.hidden_dim_c = hidden_dim_c
        self.output_dim_r = output_dim_r 
        self.output_dim_c = output_dim_c
        
        self.U1 = torch.nn.Parameter( torch.empty([hidden_dim_r, input_dim_r])  ) #Shape = [hidden_dim_r, input_dim_r]
        torch.nn.init.kaiming_uniform_(self.U1)               
        self.U1.requires_grad_()
        self.V1 = torch.nn.Parameter( torch.empty([input_dim_c, hidden_dim_c])  ) #Shape = [input_dim_c, hidden_dim_c]
        torch.nn.init.kaiming_uniform_(self.V1)
        self.V1.requires_grad_()
        self.B1 = torch.nn.Parameter( torch.empty([hidden_dim_r,hidden_dim_c])  ) #Shape = [hidden_dim_r, hidden_dim_c]
        torch.nn.init.kaiming_uniform_(self.B1)
        self.B1.requires_grad_()
        
        self.U2 = torch.nn.Parameter( torch.empty([output_dim_r, hidden_dim_r])  ) #Shape = [output_dim_r, hidden_dim_r]
        torch.nn.init.kaiming_uniform_(self.U2)               
        self.U2.requires_grad_()
        self.V2 = torch.nn.Parameter( torch.empty([hidden_dim_c, output_dim_c])  ) #Shape = [hidden_dim_c, output_dim_c]
        torch.nn.init.kaiming_uniform_(self.V2)
        self.V2.requires_grad_()
        self.B2 = torch.nn.Parameter( torch.empty([output_dim_r,output_dim_c])  ) #Shape = [output_dim_r, output_dim_c]
        torch.nn.init.kaiming_uniform_(self.B2)
        self.B2.requires_grad_()       

    def forward(self, inp):
               
        A = torch.matmul(torch.matmul(self.U1, inp) , self.V1 ) + self.B1
        Z = sigmoid(A)
        H = torch.matmul(torch.matmul(self.U2, Z) , self.V2 ) + self.B2
        return F.softmax(H)


# 150 input units, and 10*3 hidden layer size 

model = MNN(150,X_train.shape[1],10,4, 150,3)

lr=2e-2
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# define update function with weight decay

def update(x,y,lr):
    wd = 1e-5
    y_hat = model(x.float())
    
    loss = loss_func(y_hat, y) 
    loss.requres_grad = True
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item(),y_hat

losses = []
accuracy = []

for j in range(6000):                                           # runs for 6000 iterations (less iter. required if more hidden layers)
        los,output = update(X_train,Y_train,lr)
        correct=0
        for k in range(150):
            ma = 0
            for l in range(3):
                if output[k][l]>output[k][ma]:
                    ma=l
            if ma==y[k]:
                correct+=1
                
        losses.append(los)
        accuracy.append((correct/150)*100)


#plt.plot(accuracy)
#plt.show();
#plt.savefig(r'D:\python\img\mnn_iris\accuracy.jpg')

plt.plot(losses)
#plt.show();
plt.savefig(r'D:\python\img\mnn_iris\losses.jpg')