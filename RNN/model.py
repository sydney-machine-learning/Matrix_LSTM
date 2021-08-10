import torch
import torch.nn as nn


class SimpleRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(SimpleRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        
        # Fully connected layer
        self.fc1 = nn.Linear(10*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    
    def forward(self, x, device):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size, device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)

        out = torch.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

