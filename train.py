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
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradient at this value

# LR decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5 # minimum learning rate 

# Checkpoints
iter_num = 0
best_val_loss = 1e9

# System
device = 'cpu' # Use local GPUs
dtype = 'bfloat16' # is this right?
compile = True # might be pytorch based

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
    ix = mx.random.randint(len(data) - block_size, (batch_size, ))
    x = mx.stack([mx.array((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = mx.stack([mx.array((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

   # Omitted x.to(device) and y.to(device) because I don't really get how this part works, and if it's even necessary
   return x, y


# Try get vocab_size from dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta('vocab_size')
    print(f"Found vocab size = {meta_vocab_size} (inside {meta_path})")


# Model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout) # Keys might need to be rewritten 


if init_from == 'scratch':
    # init new model from scratch, probably not doing this 
    print("Initializing new model from scratch...")
    # determine the vocab size used:
    if meta_vocab_size is None:
        print("Using GPT-2 default vocab size of 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    print(f"Initializing model with OpenAI GPT-2 weights: {init_from}")
    # Initalize with GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # Read off the created config params, store in checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) 




