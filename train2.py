from model import GPTConfig, GPT

import numpy as np
import mlx
import mlx.nn as nn
import mlx.core as mx

"""
1. Make model and initialize with weights
2. Initialize optimizer and hyperparameters using optimizer (Adam) and mx.eval(model.parameters())
3. Initialize data, make the batching function, find the tokenizer (write separate script?) (do that after, i have the given vocabulary and weights and etc to work with for now
4. Set up training loop, loss functions, evaluation, etc
5. Save model weights, train
"""

MODEL = "~/rtty/code/mlxGPT/gpt-2/models/124M/model.ckpt.meta"
WEIGHTS = "~rtty/code/mlxGPT/gpt-2/models/124M/model.ckpt.data-00000-of-00001"

# --- Hyperparameters --------------------------------------------------------------------------------------------------"
# Model 
n_vocab = 50257
n_ctx = 1024 # context length
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.0

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5*8
batch_size = 12
block_size = 1024

# Optimizer
lr = 1e-4
decay_lr = True
weight_decay = 1e-1
grad_clip = 1.0
bias = False

# Training loop details
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5


# --- Model -------------------------------------------------------------------------------------------------------------"
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=n_vocab, dropout=dropout)
model = GPT(GPTConfig(**model_args)) # ** passes the dictionary into config

print(model)
