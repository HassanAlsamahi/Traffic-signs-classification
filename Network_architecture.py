import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super(Network,self).__init__()
    #Convolutional Layers
    self.conv1 = nn.Conv2d(3,20,3,padding=1)
    self.conv2 = nn.Conv2d(20,40,3,padding=1)

    #Max Pooling Layers
    self.pool = nn.MaxPool2d(2,2)

    #Linear Fully connected layers
    self.fc1 = nn.Linear(40*5*5,200)
    self.fc2 = nn.Linear(200,43)

    #Dropout
    self.dropout = nn.Dropout(p=0.2)


  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))

    x = x.view(-1,40*5*5)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
