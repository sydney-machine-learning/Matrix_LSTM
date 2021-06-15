import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1+torch.exp(x))







class MNN(nn.Module):
    
    def __init__(self, input_dim_r,input_dim_c,hidden_dim_r, hidden_dim_c,hidden_dim2_r, hidden_dim2_c,output_dim_r,output_dim_c):
	    
        super().__init__()
        
        self.input_dim_r = input_dim_r
        self.input_dim_c = input_dim_c 
        self.hidden_dim_r= hidden_dim_r
        self.hidden_dim_c = hidden_dim_c
        self.output_dim_r = output_dim_r 
        self.output_dim_c = output_dim_c
        self.hidden_dim2_r=hidden_dim2_r
        self.hidden_dim2_c=hidden_dim2_c
        
        self.U1 = torch.nn.Parameter( torch.empty([hidden_dim_r, input_dim_r])  ) #Shape = [hidden_dim_r, input_dim_r]
        torch.nn.init.kaiming_uniform_(self.U1)               
        self.U1.requires_grad_()
        self.V1 = torch.nn.Parameter( torch.empty([input_dim_c, hidden_dim_c])  ) #Shape = [input_dim_c, hidden_dim_c]
        torch.nn.init.kaiming_uniform_(self.V1)
        self.V1.requires_grad_()
        self.B1 = torch.nn.Parameter( torch.empty([hidden_dim_r,hidden_dim_c])  ) #Shape = [hidden_dim_r, hidden_dim_c]
        torch.nn.init.kaiming_uniform_(self.B1)
        self.B1.requires_grad_()
        
        self.U2 = torch.nn.Parameter( torch.empty([hidden_dim2_r, hidden_dim_r])  ) #Shape = [hidden_dim2_r, hidden_dim_r]
        torch.nn.init.kaiming_uniform_(self.U2)               
        self.U2.requires_grad_()
        self.V2 = torch.nn.Parameter( torch.empty([hidden_dim_c, hidden_dim2_c])  ) #Shape = [hidden_dim_c, hidden_dim2_c]
        torch.nn.init.kaiming_uniform_(self.V2)
        self.V2.requires_grad_()
        self.B2 = torch.nn.Parameter( torch.empty([hidden_dim2_r,hidden_dim2_c])  ) #Shape = [hidden_dim2_r, hidden_dim2_c]
        torch.nn.init.kaiming_uniform_(self.B2)
        self.B2.requires_grad_()       
        
        self.U3 = torch.nn.Parameter( torch.empty([output_dim_r, hidden_dim2_r])  ) #Shape = [output_dim_r, hidden_dim2_r]
        torch.nn.init.kaiming_uniform_(self.U3)               
        self.U3.requires_grad_()
        self.V3 = torch.nn.Parameter( torch.empty([hidden_dim2_c, output_dim_c])  ) #Shape = [hidden_dim2_c, output_dim_c]
        torch.nn.init.kaiming_uniform_(self.V3)
        self.V3.requires_grad_()
        self.B3 = torch.nn.Parameter( torch.empty([output_dim_r,output_dim_c])  ) #Shape = [output_dim_r, output_dim_c]
        torch.nn.init.kaiming_uniform_(self.B3)
        self.B3.requires_grad_()      

    def forward(self, inp):
               
        A = torch.matmul(torch.matmul(self.U1, inp) , self.V1 ) + self.B1
        Z = sigmoid(A)
        H = torch.matmul(torch.matmul(self.U2, Z) , self.V2 ) + self.B2
        Z2 = sigmoid(H)
        H2 = torch.matmul(torch.matmul(self.U3, Z2) , self.V3 ) + self.B3
        return H2
    
class Network(nn.Module):
    
    def __init__(self, input_dims, hidden_dims,hidden_dims2, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_dims2 = hidden_dims2
        self.output_dims = output_dims
        self.cnn = MNN(input_dims[0],input_dims[1], hidden_dims[0],hidden_dims[1],hidden_dims2[0], hidden_dims2[1], output_dims[0], output_dims[1])
        
    def forward(self, X):
        X = self.cnn(X)
        return F.softmax(X);
