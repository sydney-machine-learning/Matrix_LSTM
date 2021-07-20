import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MFNN(nn.Module):
    
    def __init__(self, input_dim,hidden_dim, output_dim):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.U = nn.Parameter( torch.empty([self.output_dim, self.input_dim])  ) #Shape = [output_dim, input_dim]
        nn.init.kaiming_uniform_(self.U)               
        self.U.requires_grad_()
        
        self.V = nn.Parameter( torch.empty([self.output_dim, self.hidden_dim])  ) #Shape = [output_dim, input_dim]
        nn.init.kaiming_uniform_(self.V)               
        self.V.requires_grad_()
        
        self.B = nn.Parameter( torch.empty([self.output_dim,1])  ) #Shape = [output_dim, 1]
        nn.init.kaiming_uniform_(self.B)
        self.B.requires_grad_()
        
    def forward(self, X, H):
        
        Y = torch.matmul(self.U , X ) + torch.matmul(self.V , H ) + self.B
        return Y
    
    
class RNNcell(nn.Module):
    
    def __init__(self, input_dim,hidden_dim, output_dim):
        
        super().__init__()


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.W = MFNN(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = self.hidden_dim)
        
        self.OW = nn.Parameter( torch.empty([self.output_dim, self.hidden_dim])  ) #Shape = [output_dim, hidden_dim]
        nn.init.kaiming_uniform_(self.OW)               
        self.OW.requires_grad_()
        
        self.OB = nn.Parameter( torch.empty([self.output_dim,1])  ) #Shape = [output_dim, 1]
        nn.init.kaiming_uniform_(self.OB)
        self.OB.requires_grad_()
        
    def forward(self, X, H_prev):
        
        hidden_sequence = []
        batch_size, sequence_size = 1,list(X.size())[1]
        for t in range(sequence_size):
            X_t = X[:,t]
            
            H_t = torch.tanh(self.W(X_t, H_prev))   
        
            O_t = torch.matmul(self.OW,H_t) + self.OB
           
            H_prev = H_t
            # currently O_t is of 2 dims only [batch_size,feature]
            #adds extra dimension [sequence, batch size, feature]
            hidden_sequence.append(O_t.unsqueeze(0))
        
        #concatenates along dimension 0
        #https://pytorch.org/docs/stable/generated/torch.cat.html
        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0, 1).contiguous()
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        return hidden_sequence, H_t

class Network(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
       
        self.input_dim = input_dim
        self.hidden_dim =  hidden_dim
        self.output_dim= output_dim
        self.rnn = RNNcell(input_dim,hidden_dim, hidden_dim)
        
    def forward(self, X):
        
        batch_size,seq_size = list(X.size())[0],list(X.size())[1]
        H_t = torch.zeros(tuple([batch_size, self.hidden_dim, 1]))
        X = X.unsqueeze(-1)
        X, H_t = self.rnn(X, H_t)
       
        linear = nn.Linear(in_features = self.hidden_dim*seq_size,out_features = self.output_dim)
        
        output = linear(torch.flatten(X, 1))
        return output
