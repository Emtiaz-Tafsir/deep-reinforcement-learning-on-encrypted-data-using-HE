# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:28:53 2021

@author: Lord Tafaius
"""
import agents as agn

user = agn.User(mode='plain')
platform = agn.Cloud()
platform.set_parameters(EPISODE = 30,
                       BATCH_SIZE = 10, 
                       GAMMA = 0.99, 
                       EPS_START = 0.9,
                       EPS_END = 0.05, 
                       EPS_DECAY = 300, 
                       TARGET_UPDATE = 10
                       )
user.hook(platform)
user.begin_operation()
