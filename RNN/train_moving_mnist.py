import argparse
import os
import time

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from data import MovingMNIST
from model import SimpleRNN

# Check Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU")
else:
    device =  torch.device("cpu")
    print("Training on CPU")

# Argument Parser
parser = argparse.ArgumentParser(description="Train simple RNN model on Moving MNIST dataset")
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--root-dir', type=str, default=os.getcwd())
parser.add_argument('--data-path', type=str, default=None)

# Get Args
args = parser.parse_args()


def train_model(model, train_loader, test_loader, num_epochs, optimizer, criterion, writer):
    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        
        for _, input_tensor, output_tensor in train_loader:

            optimizer.zero_grad()

            input_tensor = input_tensor.to(device)
            output_tensor = torch.squeeze(output_tensor.to(device))

            out_prime, hid = model(input_tensor, device)

            train_loss = criterion(output_tensor, out_prime)
        
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

            # create grid of images
            ind = np.random.randint(len(output_tensor))
            out_prime_reshape = torch.reshape(out_prime[ind], shape=(1, 64, 64))
            output_tensor_reshape = torch.reshape(output_tensor[ind], shape=(1, 64, 64))
            img_grid = torch.cat((out_prime_reshape, output_tensor_reshape), dim=2)

            # write to tensorboard
            writer.add_image('model_output_train', img_grid)
        
        with torch.no_grad():
            for _, input_tensor, output_tensor in test_loader:

                input_tensor = input_tensor.to(device)
                output_tensor = torch.squeeze(output_tensor.to(device))

                out_prime, hid = model(input_tensor, device)

                val_loss = criterion(output_tensor, out_prime)

                test_losses.append(val_loss.item())
            
            # create grid of images
            ind = np.random.randint(len(output_tensor))
            out_prime_reshape = torch.reshape(out_prime[ind], shape=(1, 64, 64))
            output_tensor_reshape = torch.reshape(output_tensor[ind], shape=(1, 64, 64))
            img_grid = torch.cat((out_prime_reshape, output_tensor_reshape), dim=2)

            # write to tensorboard
            writer.add_image('model_output_test', img_grid)
        
        print(f'Epoch: {epoch+1:3} loss: {train_loss.item():.4f} val loss: {val_loss.item():.4f}')
    
    return model




if __name__=='__main__':

    # Parameters
    root_dir = args.root_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Data Path 
    if args.data_path is None:
        data_path = os.path.join(root_dir, 'data', 'Moving_MNIST')
    
    # Train loader
    mm_train = MovingMNIST(root=data_path, is_train=True, n_frames_input=10, n_frames_output=1, num_objects=[2], flatten=True)
    train_loader = torch.utils.data.DataLoader(dataset=mm_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Test Loader
    mm_test = MovingMNIST(root=data_path, is_train=False, n_frames_input=10, n_frames_output=1, num_objects=[2], flatten=True)
    test_loader = torch.utils.data.DataLoader(dataset=mm_test, batch_size=args.batch_size, shuffle=False, num_workers=0)


    # Model
    model = SimpleRNN(input_size=4096, output_size=4096, hidden_dim=512, n_layers=5)
    model.to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    # Train model
    ts = str(int(time.time()))
    writer = SummaryWriter(os.path.join(root_dir, 'RNN_Arpit', 'runs', ts))
    model = train_model(model, train_loader, test_loader, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, writer=writer)

    # Save model
    model_dir = os.path.join(root_dir, 'RNN_Arpit', 'models')
    model_path = os.path.join(model_dir, ts)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)



    



