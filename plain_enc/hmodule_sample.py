# -*- coding: utf-8 -*-
"""
Created on Sat May  1 07:09:57 2021

@author: Lord Tafaius
"""
import hmodule as hm
import torch
import torch.nn as nn
import torch.optim as optim
import tenseal as ts
from time import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = hm.HConv2d(1, 4, 7, 3)
        def conv2d_size_out(size, kernel_size = 7, stride = 3):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(30)
        convh = conv2d_size_out(30)
        linear_input_size = convw * convh * 4
        self.fc1 = hm.HLinear(linear_input_size, 64)
        self.fc2 = hm.HLinear(64, 2)

    def forward(self, x, y):
        x = self.conv1(x,y)
        x.square_()
        x = self.fc1(x)
        x.square_()
        x = self.fc2(x)
        return x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
    
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

kernel_shape = net.conv1.kernel_size
stride = net.conv1.stride[0]
    
data = torch.Tensor(1, 1, 30, 30)
print()
st = time()
x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(30, 30).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
print("Encrypted Image timestamp = {} seconds".format(time()-st))
# Encrypted evaluation
enc_output = net(x_enc, windows_nb)
print("Evaluated. timestamp = {} seconds".format(time()-st))
# Decryption of result
output = enc_output.decrypt()
print("Decrypted result. timestamp = {} seconds".format(time()-st))
output = torch.tensor(output).view(1, -1)
print(output)
print(output.shape)
print(output.max(1)[1].view(1, 1).item())

