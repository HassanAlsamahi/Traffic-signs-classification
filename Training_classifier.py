import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as dataloader
from torch.utils.data import SubsetRandomSampler
import Network_architecture as network
import torch.optim as optim
import dataset as main
import torch.nn as nn
import argparse
import tempfile

parser = argparse.ArgumentParser(description='Training Traffic Sign Classifier')
parser.add_argument("-ep","--epochs", type=int, default=20, metavar='N',help='Number of training iterations (default = 20)')
parser.add_argument("--lr", type=float, default = 0.03,metavar='N',help='Learning rate')
parser.add_argument("-b","--batch_size", type=int, default=64, metavar='N', help='Batch Size(default=64)')
parser.add_argument("--model", type=str, help=("Train a pretrained model"))
parser.add_argument("--save",type=str,default= "new_model.pt",help="Name the model to save")
args = parser.parse_args()
batch_size = args.batch_size
train_loader,valid_loader,test_loader = main.loader(batch_size)
gpu_available = torch.cuda.is_available()

######### Initialize the model ###############
model = network.Network()
if args.model is not None:
    model_loaded = args.model
    model.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))
    model_saved = model
    print('Model Loaded is ',model_saved)
else:
    model_saved  = args.save

if gpu_available:
    model = model.cuda()

############ Cost function #############
criterion = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(),lr = 0.03)

Training_loss = []
Valid_loss = []
Epochs = []



###################### Training the network ######################

print("Training the network")
def train():
    valid_loss_min = np.Inf
    for epoch in range (1,args.epochs+1):
      train_loss = 0
      valid_loss = 0
      model.train()
      batch_check = 0   #checking that the model is training its first epoch
      for data,target in train_loader:
        if gpu_available:
          data,target = data.cuda(),target.cuda()
        if epoch ==1:
          batch_check +=1
          print("Batch Check= ",batch_check)

        optim.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optim.step()
        train_loss += loss.item()*data.size(0)

      #########################
      ###### Validate #########
      model.eval()
      for data,target in valid_loader:
        if gpu_available:
          data,target = data.cuda(),target.cuda()

        output = model(data)
        loss = criterion(output,target)
        valid_loss += loss.item()*data.size(0)

      train_loss = train_loss/len(train_loader.dataset)
      valid_loss = valid_loss/len(train_loader.dataset)

      print("Epoch {}.....Train Loss = {:.6f}....Valid Loss = {:.6f}".format(epoch,train_loss,valid_loss))
      Training_loss.append(train_loss)
      Valid_loss.append(valid_loss)
      Epochs.append(epoch)

      if valid_loss < valid_loss_min:
        torch.save(model.state_dict(), model_saved)
        print("Valid Loss min {:.6f} >>> {:.6f}".format(valid_loss_min, valid_loss))
        print("Model saved")
        valid_loss_min = valid_loss
    return Epochs,Training_loss,Valid_loss

Epochs,Training_loss,Valid_loss = train()
plt.plot(Epochs,Training_loss,'r',Epochs,Valid_loss,'g')
plt.show()
