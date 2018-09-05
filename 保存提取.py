# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:06:17 2018

@author: Byshev
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
#x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
            torch.nn.Linear(1,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1)
            )
    
    optimizer = torch.optim.SGD(net1.parameters(),lr=0.5)
    loss_func = torch.nn.MSELoss()
    
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#    plt.show()
    
    torch.save(net1,'net.pkl') #entire net
    torch.save(net1.state_dict(),'net_params.pkl')  #parameters
    
def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
        )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()
        
save()
restore_net()
restore_params()
        