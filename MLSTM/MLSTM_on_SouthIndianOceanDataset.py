import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

#Replace the datafile address here accordingly 
train_df = pd.read_csv(r'./data/south_indian_ocean_processedtrain1985-2001.txt', delimiter = ",", header = None, names =['id', 'date', 'longitude', 'latitude', 'speed']);
test_df = pd.read_csv(r'./data/south_indian_ocean_processedtest2006-2013.txt', delimiter = ",", header = None, names =['id', 'date', 'longitude', 'latitude', 'speed']);
 
#The 2 lists below will be used for plotting the graphs
act=[]                         
predf=[]
for i in range(0, 6):
    act.append([float(train_df['longitude'][i]),float(train_df['latitude'][i])])
    predf.append([float(train_df['longitude'][i]),float(train_df['latitude'][i])])

scaler = MinMaxScaler(feature_range=(-1, 1))
train_df['longitude'] = pd.to_numeric(train_df['longitude'], downcast="float")
train_df['latitude'] = pd.to_numeric(train_df['latitude'], downcast="float")
train_df[['longitude','latitude']] = scaler.fit_transform(train_df[['longitude','latitude']])
train_data = []
num_sample = 7    #6 datapoints + 1 prediction value

for i in range(0, train_df.shape[0]-num_sample):
    if train_df['id'][i]==train_df['id'][i+num_sample-1]:
        curr=[]
        for j in range(num_sample):
            curr.append([float(train_df['longitude'][i+j]),float(train_df['latitude'][i+j])])
        train_data.append(curr)
            
    
            
test_data = []

for i in range(0, test_df.shape[0]-num_sample):
     if test_df['id'][i]==test_df['id'][i+num_sample-1]:
        curr=[]
        for j in range(num_sample):
            curr.append([float(test_df['longitude'][i+j]),float(test_df['latitude'][i+j])])
        test_data.append(curr)


train_data = np.array(train_data)
test_data = np.array(test_data)
data_size=6
X_train=train_data[:,:data_size]
Y_train=train_data[:,data_size]
X_test=test_data[:,:data_size]
Y_test=test_data[:,data_size]



X_train, X_test, Y_train, Y_test =map(torch.FloatTensor, (X_train, X_test, Y_train, Y_test))
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
    
trainset = data_set(X_train[:1024],Y_train[:1024])        #Although the dataset has 7k+ sequences, I am using around 1k here to cut down training time
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False)          

valset = data_set(X_test[:512],Y_test[:512])
valloader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False)


class MFNN(nn.Module):
    
    def __init__(self, input_dims,hidden_dims, output_dims):
        
        super().__init__()
        
        self.input_dim_r=input_dims[0]
        self.input_dim_c=input_dims[1]
        self.hidden_dim_r=hidden_dims[0]
        self.hidden_dim_c=hidden_dims[1]
        self.output_dim_r=output_dims[0]
        self.output_dim_c=output_dims[1]
        
        self.U1 = nn.Parameter( torch.empty([self.output_dim_r, self.input_dim_r])  ) #Shape = [output_dim_r, input_dim_r]
        nn.init.xavier_uniform_(self.U1)               
        self.U1.requires_grad_()
        self.V1 = nn.Parameter( torch.empty([self.input_dim_c, self.output_dim_c])  ) #Shape = [input_dim_c, output_dim_c]
        nn.init.xavier_uniform_(self.V1)
        self.V1.requires_grad_()
        
        self.U2 = nn.Parameter( torch.empty([self.output_dim_r, self.hidden_dim_r])  ) #Shape = [output_dim_r, hidden_dim_r]
        nn.init.xavier_uniform_(self.U2)               
        self.U2.requires_grad_()
        self.V2 = nn.Parameter( torch.empty([self.hidden_dim_c, self.output_dim_c])  ) #Shape = [hidden_dim_c, output_dim_c]
        nn.init.xavier_uniform_(self.V2)
        self.V2.requires_grad_()
        
        self.B = nn.Parameter( torch.empty([self.output_dim_r,self.output_dim_c])  ) #Shape = [output_dim_r, output_dim_c]
        nn.init.xavier_uniform_(self.B)
        self.B.requires_grad_()
        
    def forward(self, X, H):
        Y = torch.matmul(torch.matmul(self.U1, X) , self.V1 ) + torch.matmul(torch.matmul(self.U2, H) , self.V2 ) + self.B
        
        
        return Y
    
    
class MLSTMlayer(nn.Module):
    
    def __init__(self, input_dims,hidden_dims, output_dims):
        
        super().__init__()
        #output_dims = hidden dims for a single layer LSTM
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        
        self.weights = nn.ModuleDict(
            {
                'input' : MFNN(input_dims = self.input_dims, hidden_dims = self.hidden_dims, output_dims = self.output_dims), 
                'forget' : MFNN(input_dims = self.input_dims, hidden_dims = self.hidden_dims, output_dims = self.output_dims),
                'output' : MFNN(input_dims = self.input_dims, hidden_dims = self.hidden_dims, output_dims = self.output_dims),
                'C_hat'  : MFNN(input_dims = self.input_dims, hidden_dims = self.hidden_dims, output_dims = self.output_dims)
                
                                   
                   
                   
               })
        
    def forward(self, X, H_prev, C_prev):
        
        hidden_sequence = []
        batch_size, sequence_size = list(X.size())[0],list(X.size())[1]
        for t in range(sequence_size):
            X_t = X[:,t]
            X_t=X_t.unsqueeze(1)
            I_t = torch.sigmoid(self.weights['input'](X_t, H_prev))   
            F_t = torch.sigmoid(self.weights['forget'](X_t, H_prev))
            C_hat_t =  torch.tanh(self.weights['C_hat'](X_t, H_prev))
            O_t = torch.sigmoid(self.weights['output'](X_t, H_prev))
        
            C_t = torch.matmul(F_t,C_prev) + torch.matmul(I_t,C_hat_t)
            H_t = torch.matmul(O_t,torch.tanh(C_t))
            H_prev = H_t
            C_prev=C_t
            # currently H_t is of 2 dims only [batch_size,feature]
            #adds extra dimension [sequence, batch size, feature]
            hidden_sequence.append(H_t.unsqueeze(0))
        
        #concatenates along dimension 0
        #https://pytorch.org/docs/stable/generated/torch.cat.html
        hidden_sequence = torch.cat(hidden_sequence, dim=0)
        hidden_sequence = hidden_sequence.transpose(0, 1).contiguous()
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        return hidden_sequence, H_t, C_t

class Network(nn.Module):
    
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
       
        self.input_dims = input_dims
        self.hidden_dims =  hidden_dims
        self.output_dims = output_dims
        self.lstm1 = MLSTMlayer(input_dims,hidden_dims, hidden_dims)
        
    def forward(self, X):
      
        batch_size = list(X.size())[0]
        H_t,C_t = (torch.zeros(tuple([batch_size, *self.hidden_dims])),                 #H_t => Hidden States , C_t = Cell states      
                        torch.zeros(tuple([batch_size,*self.hidden_dims])))
    
        X, H_t, C_t = self.lstm1(X, H_t, C_t)
        p = nn.Linear(in_features = list(X.size())[1]*self.hidden_dims[0]*self.hidden_dims[1],out_features = 2)  #2 output features for longitude and latitude
        Y=torch.empty((batch_size,2))
        for i in range(batch_size):
            temp=torch.flatten(X[i])
            Y[i] = p(temp)
        
        return Y

input_dims=[1,2]
hidden_dims = [10,10]

model = Network(input_dims, hidden_dims, hidden_dims) 
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 50
for i in range(epochs):
    for seq, labels in trainloader:
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()

    if i%5 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

predictions=[]
with torch.no_grad():
    for seq, labels in trainloader:
        y_pred = model(seq)
        y_np=np.array(y_pred)
        lab_np=np.array(labels)
        y_np=scaler.inverse_transform(y_np)
        lab_np=scaler.inverse_transform(lab_np)
        y_np = y_np.tolist()
        lab_np = lab_np.tolist()
        for j in range(15):                       #This loop is to feed the first 15 values to the lists used for graphing
            act.append(lab_np[j])
            predf.append(y_np[j])
            
        #print(f'pred_long:{y_np[:,0]}    pred_lat:{y_np[:,1]}   label_long:{lab_np[:,0]}    label_lat:{lab_np[:,1]}')  
        #Uncomment the above line to see the predictions
        

fig = plt.figure()
lat_act=[]
long_act=[]
lat_predf=[]
long_predf=[]
for i in range(16):
    long_act.append(act[i][0]) #Actual Longitude
    lat_act.append(act[i][1])  #Actual Latitude
    long_predf.append(predf[i][0]) #Predicted Longitude
    lat_predf.append(predf[i][1])  #Predicted Latitude
plt.scatter(lat_act, long_act)
plt.plot(lat_act, long_act, '.b-') 
plt.scatter(lat_predf, long_predf)
plt.plot(lat_predf, long_predf, 'xr-') 
#plt.savefig(r'D:\python\img\MLSTM_standardised_withbatching\010.png')  
#Uncomment the above line to save the plot. Change the destination folder accordingly.
plt.show()
