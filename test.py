import torch
import os
import numpy as np
import mlx.core as mx
data = np.memmap(os.path.join('data', 'train.bin'), dtype=np.uint16, mode='r')

block_size=1024
batch_size=12
ix2 = torch.randint(len(data) - block_size, (batch_size,))
print(type(ix2[0]))
ix = mx.random.randint(low=0, high=len(data)-block_size, shape=(batch_size,))   
print(ix[0])
print(type(ix[0]))
print(type(ix[0].item()))
x = mx.ones(3)
print(x.shape)
