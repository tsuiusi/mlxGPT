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
x = torch.stack([torch.from_numpy((data[i.item():i.item()+block_size]).astype(np.int64)) for i in ix])
print(x)
x2 = mx.stack([mx.array((data[i:i+block_size]).astype(np.int64)) for i in ix])
print(x2)
