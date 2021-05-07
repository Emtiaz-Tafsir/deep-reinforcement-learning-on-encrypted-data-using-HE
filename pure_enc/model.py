# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:08:38 2021

@author: Lord Tafaius
"""

import hmodule as hm
import torch
import torch.nn as nn
import torch.optim as optim
import tenseal as ts
import random
from collections import deque
from collections.abc import Iterable

class Network(nn.Module):
    def __init__(self, in_height, in_width, output_n, mode='plain'):
        super(Network, self).__init__()
        def conv2d_size_out(size, kernel = 7, stride = 3):
            return (size - (kernel - 1) - 1) // stride  + 1
        convw = conv2d_size_out(in_width)
        convh = conv2d_size_out(in_height)
        linear_input_size = convw * convh * 4
        if not (mode=='plain' or mode=='encrypted'):
            raise ValueError('Unknown Mode: '+mode)
        self.conv1e = hm.HConv2d(in_feature=1, out_feature=4, kernel=7, mode=mode, stride=3)
        self.fc1e = hm.HLinear(linear_input_size, 64, mode=mode)
        self.fc2e = hm.HLinear(64, output_n, mode=mode)
        self.mode=mode
        self.linear_input_size=linear_input_size
    
    def _pforward(self, x):
        x = self.conv1e(x)
        x=x**2
        x = x.view(-1, self.linear_input_size)
        x = self.fc1e(x)
        x=x**2
        x = self.fc2e(x)
        return x
        
    def _hforward(self, x, y, z):
        x = self.conv1e(x,y,z)
        x1=x[0]**2
        x1 = x1.view(-1, self.linear_input_size)
        x2 = x[1].square_()
        x = (x1,x2)
        x = self.fc1e(x)
        x1=x[0]**2
        x2 = x[1].square_()
        x = (x1,x2)
        x = self.fc2e(x)
        return x
    
    def forward(self, x, window=0, mean=0.00005):
        if isinstance(x, tuple):
            outlist=[]
            inlist =[]
            for img, win, m in x:
                r = torch.full((1,1,30,30),m)
                inlist.append(r)
                out=self._bforward(img, win)
                outlist.append(out)
            c = torch.cat(inlist)
            self.conv1e.mode='plain'
            self.fc1e.mode='plain'
            self.fc2e.mode='plain'
            ca = self._pforward(c)
            self.conv1e.mode='encrypted'
            self.fc1e.mode='encrypted'
            self.fc2e.mode='encrypted'
            return ca , outlist
        else:
            return self._forward(x,window,mean)
        
        
    def _forward(self, x, window=0, mean=0.00005):
        if self.mode=='encrypted':
            return self._hforward(x, window, mean)
        else:
            return self._pforward(x)
            
    def _bforward(self, x, y):
        x = self.conv1e(x,y,batch=True)
        x.square_()
        x = self.fc1e(x,batch=True)
        x.square_()
        x = self.fc2e(x,batch=True)
        return x
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class ReplayMemory(object):

    def __init__(self, capacity, transition):
        self.memory = deque([],maxlen=capacity)
        self.Transition = transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)