import os
import time
import math
import pickle 
from contextlib import nullcontext

import numpy as np
import mlx
import mlx.nn as nn
import mlx.core as mx

from model import GPTConfig, GPT

"""
1. Initialize hyperparameters
2. Load data
3. Load model
4. Training loop
5. Save weights
"""

# Hyperparameters
# ------------------------------------------------------------------------------
# Default values for GPT-2
# I/O
out_dir = 'out'
eval_interval = 2000
eval_iters = 200

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# AdamW
lr = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # Clip gradient at this value

# LR decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5 # Minimum learning rate 

# Tokens
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter)")

# ------------------------------------------------------------------------------ 
# Poor man's data loader 
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.r
