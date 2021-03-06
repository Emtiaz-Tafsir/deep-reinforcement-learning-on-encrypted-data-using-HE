# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:28:53 2021

@author: Lord Tafaius
"""
import agents as agn

user = agn.User()
platform = agn.Cloud()
platform.set_parameters(EPISODE = 50,
                       BATCH_SIZE = 128, 
                       GAMMA = 0.99,
                       EPS_START = 0.9,
                       EPS_END = 0.05, 
                       EPS_DECAY = 400,
                       TARGET_UPDATE = 10
                       )
user.hook(platform)
#user.begin_operation()
model_path = 'prev_models/_current/model.pth'
context_path = 'prev_models/_current/context.bytes'
user.send_models_to_cp(model_path, context_path, EP=50)

