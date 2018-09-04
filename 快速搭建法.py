# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:11:12 2018

@author: Byshev
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# method1
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net1 = Net(1 ,10, 1) 

# method2
net2 = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1))

print(net1)
print(net2)