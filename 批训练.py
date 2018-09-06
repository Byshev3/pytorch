# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:01:03 2018

@author: Byshev
"""

import torch
import torch.utils.data as Data

Batch_size = 7
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

#先转换成torch能识别的Dataset
torch_dataset = Data.TensorDataset(x,y)

#把dataset放入DataLoader
loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = Batch_size,
        shuffle = False,
        num_workers = 0,
        )

for epoch in range(3):
    for step, (batch_x,batch_y) in enumerate(loader):
        print('Epoch:', epoch, '|Step:', step, '|batch x:', batch_x.numpy(), '|batch y:', batch_y.numpy())
        
        
        