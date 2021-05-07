# -*- coding: utf-8 -*-
"""
Created on Sat May  1 06:09:42 2021

@author: Lord Tafaius
"""
import torch
import torch.nn as nn
import tenseal as ts

class HConv2d(nn.Conv2d):
    
    def __init__(self, in_feature, out_feature, kernel, stride=1):
        super(HConv2d, self).__init__(in_feature, out_feature, kernel, stride)
        self.weight = torch.nn.Parameter(data=torch.Tensor(out_feature,in_feature, kernel, kernel), requires_grad=True)
        self.weight.data.uniform_(-1, 1)
    
    def forward(self, inp, windows_nb):
        self.weight_as_list = self.weight.data.view(
            self.out_channels, self.kernel_size[0],
            self.kernel_size[1]
        ).tolist()
        self.bias_as_list = self.bias.data.tolist()
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.weight_as_list, self.bias_as_list):
            y = inp.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        return ts.CKKSVector.pack_vectors(enc_channels)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class HLinear(nn.Linear):
    
    def __init__(self, in_features, out_features):
        super(HLinear,self).__init__(in_features, out_features)
        self.weight = torch.nn.Parameter(data=torch.Tensor(out_features, in_features), requires_grad=True)
        self.weight.data.uniform_(-1, 1)
        
    def forward(self, inp):
        self.weight_as_list = self.weight.T.data.tolist()
        self.bias_as_list = self.bias.data.tolist()
        return inp.mm(self.weight_as_list) + self.bias_as_list
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        