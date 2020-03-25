#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import cv2, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import ToPILImage
show=ToPILImage()
import numpy as np
import matplotlib.pyplot as plt
from DataLoader_ILSVRC import ILSVRC2012
from functools import partial 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def imshow(img):
    cv2.imshow("",img)
    cv2.waitKey(20)
    # cv2.destroyAllWindows()

class LocalResponseNorm(nn.Module):
    __constants__ = ['size', 'alpha', 'beta', 'k']
    
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2.):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return F.local_response_norm(input, self.size, self.alpha, self.beta,
                                     self.k)

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x, 1)
        
class ZFNet(nn.Module):
    def __init__(self, num_classes):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            LocalResponseNorm(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)
        )
        self.flatten = Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ) 
        self.feature_maps = dict()
        self.pool_locs = dict()
    def init_weights(self):
        def init_function(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                m.bias.data.fill_(0)    
        _ = self.apply(init_function)
    def store(self):
        def hook(module, input, output, key):
            if isinstance(module, nn.MaxPool2d):
                self.feature_maps[key] = output[0]
                self.pool_locs[key] = output[1]
            else:
                self.feature_maps[key] = output
        for idx, layer in enumerate(self.features):
            layer.register_forward_hook(partial(hook, key=idx))
    
    def forward(self, x):        
        self.feature_maps = dict()
        self.pool_locs = dict()
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, _ = layer(x)
            else:
                x = layer(x)
                
        x = self.flatten(x)
        x = self.classifier(x)     
        return x

train_dir = '/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train'
val_dir = '/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_val'
dirname_to_classname_path = 'dirname_to_classname'
pretrained_weights = None
  
num_epoch = 70
batch_size_train = 128
momentum = 0.9
learning_rate = 0.01
num_classes = 100

trainset = ILSVRC2012(train_dir, dirname_to_classname_path, num_classes)
testset = ILSVRC2012(val_dir, dirname_to_classname_path, num_classes)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_train, shuffle=False, num_workers=8)

net = ZFNet(num_classes).cuda()

if pretrained_weights != None:
    net.load_state_dict(torch.load(pretrained_weights))
else:
    net.init_weights()
criterion = nn.CrossEntropyLoss()
optimizer= optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

train_loss_list = list()
train_accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()

for epoch in range(num_epoch):
    time_s = time.time()
    print('Epoch : ', epoch + 1)

    net.train()
    
    for batch_idx, (img, y_GT) in enumerate(train_dataloader):
        img = img.permute(0, 3, 1, 2).float()
         
        y_PD = net(img.cuda())
        loss = criterion(y_PD, y_GT.long().cuda())
        acc_batch = np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1))
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            print("Epoch {}, Training Data Num {}, Loss {}, Batch Accuracy {}%".format(epoch+1, (batch_idx + 1) * batch_size_train, loss.item(), np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))/len(y_GT)*100))
            print("labels(GT) = ", y_GT[:10].numpy())
            print("labels(PD) = ", np.argmax(y_PD.cpu().data.numpy()[:10], axis=1))
    
  
    net.eval()
    
    acc_train = 0
    loss_train = 0 
    for batch_idx, (img, y_GT) in enumerate(train_dataloader):
        img = img.permute(0, 3, 1, 2).float()
        with torch.no_grad():
            y_PD = net(img.cuda())
        loss = criterion(y_PD, y_GT.long().cuda())
        acc_train += np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))
        loss_train += loss.item()
    
    acc_train /= len(trainset)
    loss_train /= len(trainset) / batch_size_train
    train_loss_list.append(loss_train)
    train_accuracy_list.append(acc_train)
    print("Train Loss : ", loss_train, "Accuracy : %.2f%%" %(acc_train * 100))
    
    scheduler.step(loss_train) # adjsut learning rate. 
    
    acc_test = 0
    loss_test = 0   
    for batch_idx, (img, y_GT) in enumerate(test_dataloader):
        img = img.permute(0, 3, 1, 2).float()
        with torch.no_grad():
            y_PD = net(img.cuda())
        loss = criterion(y_PD, y_GT.long().cuda())
        acc_test += np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))
        loss_test += loss.item()
    acc_test /= len(testset)
    loss_test /= len(testset) / batch_size_train
    test_loss_list.append(loss_test)
    test_accuracy_list.append(acc_test)
    print("Test Loss : ", loss_test, "Accuracy : %.2f%%" %(acc_test * 100))
    if not os.path.isdir('./weights'):
        os.mkdir('weights')
    torch.save(net.state_dict(), 'weights/alexnet_pretrained_weights_{}.pth'.format(epoch+1))
    
    print("Time Elapsed : ", time.time() - time_s)
 
x = np.arange(len(train_accuracy_list) + 1)
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(x, [0] + train_accuracy_list)
plt.plot(x, [0] + test_accuracy_list)
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.grid(True)
plt.savefig('Accuracy_numCls{}_epoch{}.png'.format(num_classes, epoch+1)) 
plt.show()   

plt.xlabel('epochs')
plt.ylabel('Loss')
plt.plot(x, train_loss_list[0:1] + train_loss_list)
plt.plot(x, train_loss_list[0:1] + test_loss_list)
plt.legend(['training loss', 'testing loss'], loc='upper right')
plt.savefig('Loss_numCls{}_epoch{}.png'.format(num_classes, epoch+1))
plt.show()    
