import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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
