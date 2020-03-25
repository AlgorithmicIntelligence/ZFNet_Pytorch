#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import cv2, os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.fill_(0)

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
        self.store()
        
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
    

class D_ZFNet(nn.Module):
    def __init__(self, net):
        super(D_ZFNet, self).__init__()
        self.layers = list()
        for i, layer in enumerate(net.features):
            if isinstance(layer, nn.Conv2d):
                self.layers.append(nn.ConvTranspose2d(layer.out_channels, layer.in_channels, layer.kernel_size, layer.stride, layer.padding, bias=False))
                self.layers[i].weight.data = layer.weight.data
            elif isinstance(layer, nn.ReLU):
                self.layers.append(nn.ReLU(inplace=True))
            elif isinstance(layer, nn.MaxPool2d):
                self.layers.append(nn.MaxUnpool2d(layer.kernel_size, layer.stride, layer.padding))
            else:
                self.layers.append(None)
    

    
    def forward(self, x, idx_vis, net):
        if not 0 <= idx_vis < len(self.layers):
            raise ValueError('idx_vis must in the range {} - {}'.format(0, len(self.layers)-1))
        for idx in range(idx_vis, -1, -1):
            if isinstance(self.layers[idx], nn.MaxUnpool2d):
                x = self.layers[idx](x, net.pool_locs[idx], net.feature_maps[idx-1].shape)
            elif self.layers[idx] == None:
                continue
            else:
                x = self.layers[idx](x)
        return x
    
train_dir = '/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train'
dirname_to_classname_path = 'dirname_to_classname'
pretrained_weights = '/media/nickwang/StorageDisk2/Code/Architecture/ZFNet_Pytorch/weights/alexnet_pretrained_weights_100.pth'
  
num_classes = 100

trainset = ILSVRC2012(train_dir, dirname_to_classname_path, num_classes)

net = ZFNet(num_classes).cuda()
if pretrained_weights != None:
    net.load_state_dict(torch.load(pretrained_weights))
else:
    _ = net.apply(init_weights)    
net.eval()

d_net = D_ZFNet(net).cuda()
d_net.eval()

img, label = trainset.__getitem__(3000)
img_origin = (img.copy() + trainset.img_means/255.)
img = np.expand_dims(img, 0)
img = torch.from_numpy(img).float().cuda()
img = img.permute(0, 3, 1, 2).float()
with torch.no_grad():
    conv_output = net(img)    

print('GT :', label)
print('PD :', np.argmax(conv_output.cpu().numpy()))

plt.figure(figsize=(5, 3), dpi=300)
plt.subplot(3, 5, 1)
plt.title('Original Image', y=0.9, fontsize=3)
plt.axis('off')
plt.imshow(img_origin)
idx_layer = 14
feature_maps = net.feature_maps[idx_layer]
feature_sort = torch.argsort(torch.sum(feature_maps[0], axis=(-1, -2)), descending=True)
for idx_feature in range(14):
    feature_map = torch.zeros(feature_maps.shape).cuda()
    feature_map[0, feature_sort[idx_feature]] = feature_maps[0, feature_sort[idx_feature]]
    with torch.no_grad():
        img_ = d_net(feature_map, idx_layer, net)
    img_ = img_.cpu().numpy()[0].transpose(1, 2, 0)
    img_ = (img_ - img_.min()) / (img_.max() - img_.min()) * 255.
    img_ = img_.astype(np.uint8)
    plt.subplot(3 , 5, idx_feature+2)
    plt.title('Layer {}, Feature {}'.format(idx_layer+1, idx_feature+1), y=0.9, fontsize=3)
    plt.axis('off')
    plt.imshow(img_)
plt.savefig('FeatureVisualization_SameLayer.jpg')
plt.show()


plt.figure(figsize=(5, 3), dpi=300)
plt.subplot(3, 5, 1)
plt.title('Original Image', y=0.9, fontsize=3)
plt.axis('off')
plt.imshow(img_origin)
for idx_layer in range(14):
    idx_feature = 0
    feature_maps = net.feature_maps[idx_layer]
    feature_sort = torch.argsort(torch.sum(feature_maps[0], axis=(-1, -2)), descending=True)
    feature_map = torch.zeros(feature_maps.shape).cuda()
    feature_map[0, feature_sort[idx_feature]] = feature_maps[0, feature_sort[idx_feature]]
    with torch.no_grad():
        img_ = d_net(feature_map, idx_layer, net)
    img_ = img_.cpu().numpy()[0].transpose(1, 2, 0)
    img_ = (img_ - img_.min()) / (img_.max() - img_.min()) * 255.
    img_ = img_.astype(np.uint8)
    plt.subplot(3 , 5, idx_layer+2)
    plt.title('Layer {}, Feature {}'.format(idx_layer+1, idx_feature+1), y=0.9, fontsize=3)
    plt.axis('off')
    plt.imshow(img_)

plt.savefig('FeatureVisualization_DifferentLayer.jpg')
plt.show()


