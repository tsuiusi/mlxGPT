from model import GPTConfig, GPT

import os
import time
import numpy as np
import mlx
import mlx.nn as nn
import mlx.core as mx
from mlx.optimizers import AdamW

"""
1. Make model and initialize with weights
2. Initialize optimizer and hyperparameters using optimizer (Adam) and mx.eval(model.parameters())
3. Initialize data, make the batching function, find the tokenizer (write separate script?) (do that after, i have the given vocabulary and weights and etc to work with for now
4. Set up training loop, loss functions, evaluation, etc
5. Save model weights, train
"""

# MODEL = "~/rtty/code/mlxGPT/gpt-2/models/124M/model.ckpt.meta"
# WEIGHTS = "~rtty/code/mlxGPT/gpt-2/models/124M/model.ckpt.data-00000-of-00001"

# --- Hyperparameters --------------------------------------------------------------------------------------------------"
# Model 
n_vocab = 50257
n_ctx = 1024 # context length
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.0

# Data
dataset = 'data'
gradient_accumulation_steps = 5*8
batch_size = 12
block_size = 1024

# Optimizer
lr = 1e-4
decay_lr = True
weight_decay = 1e-1
grad_clip = 1.0
beta1 = 0.9
beta2 = 0.95
bias = False

# Training loop details
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# --- Data --------------------------------------------------------------------------------------------------------------"
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter}")

# Take data loaded from Karpathy's Shakespeare prepare.py
train_data = mx.array(np.memmap(os.path.join(dataset, 'train.bin'), dtype=np.uint16, mode='r'), dtype=mx.uint16)
val_data = mx.array(np.memmap(os.path.join(dataset, 'val.bin'), dtype=np.uint16, mode='r'), dtype=mx.uint16)

print(train_data[:10])
print(val_data[:10])


# --- Model  and Optimizer ----------------------------------------------------------------------------------------------"
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=n_vocab, dropout=dropout)
model = GPT(GPTConfig(**model_args)) # ** passes the dictionary into config

optimizer = AdamW(lr, (beta1, beta2), weight_decay=weight_decay)

def loss_fn(model, X, y, sample_size):
    pass

def eval_fn(model, X, y, sample_size):
    pass


