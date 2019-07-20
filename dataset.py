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
import torch.nn as nn



gpu_available = torch.cuda.is_available()
print(gpu_available)

transforms = transforms.Compose([transforms.Resize((20,20)),
                                transforms.RandomRotation(60),
                                transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])


Training_path = "Dataset/Training"
Testing_path = "Dataset/Testing"

training_dataset = datasets.ImageFolder(Training_path,transform=transforms)
testing_dataset = datasets.ImageFolder(Testing_path,transform=transforms)
classes = sorted(os.listdir(Training_path))
#print(len(classes))

##################### Loading the data ##########################
valid_size = 0.2
#test_size = 0.5
#global batch_size
#batch_size = 0
num_workers = 0


num_data = len(training_dataset)
train_indices = list(range(num_data))
np.random.shuffle(train_indices)
valid_split = int(np.floor(valid_size * num_data))
train_idx,valid_idx =train_indices[valid_split:], train_indices[:valid_split]

num_test = len(testing_dataset)
test_indices = list(range(num_test))
np.random.shuffle(test_indices)
#test_split = int(np.floor(test_size * len(valid_indices)))
test_idx = test_indices
#print(len(valid_idx))

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

def loader(batch_size):
    train_data = dataloader.DataLoader(training_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_data = dataloader.DataLoader(training_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_data = dataloader.DataLoader(testing_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    return train_data,valid_data,test_data
