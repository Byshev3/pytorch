# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:58:31 2018

@author: Byshev
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
        root = r'C:\Users\Byshev\Documents\Python Scripts\pytorch\mnist',
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download = DOWNLOAD_MNIST,
        )

#print(train_data.train_data.size())
#print(train_data.train_labels.size())
#plt.imshow(train_data.train_data[1],cmap = 'gray')
#plt.title('%d' % train_data.train_labels[1])

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28) 且自动归一化除以255
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)

test_data = torchvision.datasets.MNIST(root = r'C:\Users\Byshev\Documents\Python Scripts\pytorch\mnist', train = False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000] 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels = 1,
                          out_channels = 16,
                          kernel_size = 5,
                          stride = 1,
                          padding = 2,
                          ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                )
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        
        
            
        






