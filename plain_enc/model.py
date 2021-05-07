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
        if mode=='plain':
            self.linear_input_size = linear_input_size
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=7, stride=3)
            self.fc1 = nn.Linear(self.linear_input_size, 64)
            self.fc2 = nn.Linear(64, output_n)
        elif mode=='encrypted':
            self.conv1 = hm.HConv2d(in_feature=1, out_feature=4, kernel=7, stride=3)
            self.fc1 = hm.HLinear(linear_input_size, 64)
            self.fc2 = hm.HLinear(64, output_n)
        else:
            raise ValueError('Unknown Mode: '+mode)
        self.mode=mode
    
    def _pforward(self, x):
        x = self.conv1(x)
        x=x**2
        x = x.view(-1, self.linear_input_size)
        x = self.fc1(x)
        x=x**2
        x = self.fc2(x)
        return x
        
    def _hforward(self, x, y):
        x = self.conv1(x,y)
        x.square_()
        x = self.fc1(x)
        x.square_()
        x = self.fc2(x)
        return x
    
    def forward(self, x, window=0):
        if isinstance(x, tuple):
            outlist=[]
            for img, win in x:
                out=self._hforward(img, win)
                outlist.append(out)
            return outlist
        return self._pforward(x) if self.mode=='plain' else self._hforward(x, window)
    
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