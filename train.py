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
vocab_size = 50257
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
eval_interval = 2000
lr_decay_iters = 600000
min_lr = 6e-5
iter_num = 0 
best_val_loss = 1e9 # Big initial value 

# --- Data --------------------------------------------------------------------------------------------------------------"
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter}")

# Take data loaded from Karpathy's Shakespeare prepare.py
train_data = np.memmap(os.path.join(dataset, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(dataset, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    # Since the objective of a Transformer is to perform next-token prediction the target vector will be the input vector shifted by index 1
    data = train_data if split  == 'train' else val_data
    ix = mx.random.randint(low=0, high=len(data)-block_size, shape=(batch_size,))   
    x = mx.stack([mx.array((data[i.item():i.item()+block_size]).astype(np.int64)) for i in ix]) # .item() because the datatype inside mx.random.randint is also mx.array and not like torch.tensor (which can be used in lists) so if I access the item it'll work
    y = mx.stack([mx.array((data[i.item()+1:i.item()+block_size+1]).astype(np.int64)) for i in ix])
    # Karpathy does some CUDA-ing here 
    
    return x, y

# --- Model  and Optimizer ----------------------------------------------------------------------------------------------"
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout)
model = GPT(GPTConfig(**model_args)) # ** passes the dictionary into config

optimizer = AdamW(lr, (beta1, beta2), weight_decay=weight_decay)

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X)[0], y))

loss_function = nn.value_and_grad(model, loss_fn)

# --- Training loop -----------------------------------------------------------------------------------------------------"
X, Y = get_batch('train') # First batch, can continuously sample because it's random sampling
local_iter_num = 0
no_epochs = 200 # putting this here for now

while True:

    tic = time.perf_counter()
    loss, grads = loss_function(model, X, Y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
   
    best_val_loss = max(loss.item(), best_val_loss)	
    toc = time.perf_counter()

