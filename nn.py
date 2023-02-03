## nn.py
# Author: Yamauchi Yudai, 2022, RWTH Aachen Univ

#Model of neural network

import torch
from torch import nn
import torch.nn.functional as F

# simple NN for the paper
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(512,2048)
        self.fc2 = torch.nn.Linear(2048,1024)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
