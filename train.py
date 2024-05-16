from model import GPTConfig, GPT

import os
import time
import sys
import numpy as np
import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten, tree_map


"""
1. Make model and initialize with weights
2. Initialize optimizer and hyperparameters using optimizer (Adam) and mx.eval(model.parameters())
3. Initialize data, make the batching function, find the tokenizer (write separate script?) (do that after, i have the given vocabulary and weights and etc to work with for now
4. Set up training loop, loss functions, evaluation, etc
5. Save model weights, train
"""

# MODEL = "~/rtty/code/mlxGPT/gpt-2/models/124M/model.ckpt.meta"
# WEIGHTS = "~rtty/code/mlxGPT/gpt-2/models/124M/model.ckpt.data-00000-of-00001"
def get_dataset():
    if sys.argv[1] and (sys.argv[1] == 'shakespeare' or sys.argv[1] == 'openwebtext'):
        return sys.argv[1]
    return 'shakespeare'

# --- Hyperparameters --------------------------------------------------------------------------------------------------"
# Model 
vocab_size = 50257
n_ctx = 1024 # context length
n_embd = 768
num_heads = 12
n_layer = 12
dropout = 0.0

# Data
dataset = get_dataset()
out_dir = '~/code/mlxGPT/'
gradient_accumulation_steps = 5*8
batch_size = 12
block_size = 1024

# Optimzer
grad_clip = 1.0
beta1 = 0.9
beta2 = 0.95
bias = False

# Training loop details
learning_rate = 1e-4
decay_lr = True
weight_decay = 1e-1
warmup_iters = 2000
eval_interval = 2000
lr_decay_iters = 600000
min_lr = 6e-5
iter_num = 1 
best_val_loss = 1e9 # Big initial value 


# --- Data --------------------------------------------------------------------------------------------------------------"
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps
print(f"Tokens per iteration will be: {tokens_per_iter}")

# Take data loaded from Karpathy's Shakespeare prepare.py
train_data = np.memmap(os.path.join(dataset, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(dataset, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    # Since the objective of a Transformer is to perform next-token prediction the target vector will be the input vector shifted by index 1
    data = train_data if split  == 'train' else val_data
    ix = mx.random.randint(low=0, high=len(data)-block_size, shape=(batch_size,)) # Get random indices within the acceptable range to sample from 

    x = mx.stack([mx.array((data[i.item():i.item()+block_size]).astype(np.int64)) for i in ix]) # .item() because the datatype inside mx.random.randint is also mx.array and not like torch.tensor (which can be used in lists) so if I access the item it'll work
    y = mx.stack([mx.array((data[i.item()+1:i.item()+block_size+1]).astype(np.int64)) for i in ix]) # Next token prediction so it's shifted by 1
    # Karpathy does some CUDA-ing here 
    
    return x, y

def batch_iterate(split):
    """
    what mnist does is for a range in 0 to y.size, pick batch_size no. samples and takes their ids
    resnet does a similar thing, takes a random list of indices and yields them when done
    what get_batch does is get batch_size no. sequences from the data and returns them every time

    write a version that yields the data instead?
    """
    data = train_data if split == 'train' else val_data


# --- Model and Optimizer ----------------------------------------------------------------------------------------------"
model_args = dict(n_layer=n_layer, num_heads=num_heads, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout)
model = GPT(GPTConfig(**model_args)) # ** passes the dictionary into config
mx.eval(model.parameters())

optimizer = AdamW(learning_rate, (beta1, beta2), weight_decay=weight_decay)
mx.eval(optimizer.state)


def loss_fn(model, X, y):
    logits = model(X)
    loss = nn.losses.cross_entropy(logits, y) 
    return mx.mean(loss)

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_lr + coeff * (learning_rate - min_lr)


def step(X, y, gradient_accumulation_steps):
    accumulated_grads = tree_map(lambda x: mx.zeros_like(x), model.parameters())
    accumulated_loss = 0.0

    for micro_step in range(gradient_accumulation_steps):
        loss, grads = loss_and_grad_fn(model, X, y)

        # Scale gradients, add it to running sum
        accumulated_grads = tree_map(lambda accumulated, new: accumulated + (new / gradient_accumulation_steps), accumulated_grads, grads,)

        # Evaluate gradients
        tree_map(lambda grad: mx.eval(grad), accumulated_grads,)
        
        # Accumulate loss
        accumulated_loss += loss.item()
    
    # Average loss
    loss = mx.array(accumlated_loss / gradient_accumulation_steps)
    optimizer.update(model, accumulated_grads)

    return loss

def step2(X, y):
    loss, grad = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grad)
    return loss


def console_log(iter_num, loss, tic):
    toc = time.perf_counter()
    print(f"Iteration: {iter_num} | Loss: {loss.item():.3f} | Learning Rate: {optimizer.learning_rate.item():.3f} | Time: {(toc - tic):.3f}")

    # Reuse as tic for next cycle
    return toc

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X)) == y)

# --- Training loop -----------------------------------------------------------------------------------------------------"
no_iters = 10 # putting this here for now
save_interval = 10

X, y = get_batch('train')
tic = time.perf_counter()

print('Starting training...')

while True: 
#     lr = get_lr(iter_num) if decay_lr else learning_rate
#     optimizer.set_learning_rate(lr)
    # loss = step(X, y, gradient_accumulation_steps)
    optimizer.learning_rate = get_lr(iter_num)
    loss = step2(X, y)
    mx.eval([model.state, optimizer.state])
    X, y = get_batch('train')

    # Logging
    tic = console_log(iter_num, loss, tic)

    # Periodic saving
    if iter_num % save_interval == 0:
        valX, valY = get_batch('val')
        val_loss = loss_fn(model, valX, valY)

        best_val_loss = min(best_val_loss, val_loss.item())
        print(f'Current loss: {best_val_loss}')

        model.save_weights('gpt2.npz')

    iter_num += 1

    if iter_num > no_iters:
        break


"""
Todo:
    1. Write eval + get validation loss (i only have training loss so far)
    2. tokenizer? decoding?
    3. gradient accumulation
    5. inference
"""
