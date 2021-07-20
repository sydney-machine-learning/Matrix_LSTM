import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_file = r"D:\python\RNN\mnist_test_seq.npy"

data = np.load(data_file)
#data is of shape 20 X 10,000 X 64 X 64
data=data.swapaxes(1,0)
#Above line swaps the 20 and 10,000 dimensions.
data = data[:100]
data = data/255 #scaling 

window_size = 10
final_data = []
labels = []
for i in range(data.shape[0]):
    for j in range(20-window_size):
        final_data.append(data[i,j:j+window_size])
        labels.append(data[i,j+window_size])

final_data = np.array(final_data)
labels = np.array(labels)


X_train, X_test, Y_train, Y_test = train_test_split(final_data, labels, test_size=0.3, shuffle = True, stratify = None)

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
    
trainset = data_set(X_train,Y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)          

valset = data_set(X_test,Y_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)



