import math
from dataclasses import dataclass

import mlx
import mlx.nn as nn
import mlx.core as core

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        # I don't think I need to initialize the weights and biases because lazy eval
        self.weights = core.ones((ndims, ))
        self.bias = core.ones((ndims, )) if bias else None

    def forward(self, x):
        return nn.LayerNorm(x, self.weights, dims=ndims, eps=1e-5) 

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # K Q V for all heads in a batch
        # nn.Linear only defines the input dim, output dim, and bias
        # 3 because it's then split into Q K V
        self.c_attn = nn.Linear(config.n_embd, 3* config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        # Used to prevent overfitting, random neurons are dropped out to ensure the network learns redundant connections (more than one way to the answer) 
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention: mechanism that optimizes self-attention calculation in transformer models, improves efficiency
        # Was implemented in the original but not available in mlx yet, will add later
        self.register_buffer('bias', core.reshape(core.tril(core.ones([config.block_size, config.block_size])), [1, 1, config.bloc_size, config.block_size]))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size()
        
        # query, key, value for all heads in batch and move head forward to be the batch dim
        q, k, v = core.split(self.c_attn(x), self.n_embd, axis=2)
        k = core.transpose(core.reshape(k, [B, T, self.n_head, C // self.n_head]), axes=[1, 2])
        q = core.transpose(core.reshape(q, [B, T, self.n_head, C // self.n_head)], axes=[1, 2])
        v = core.transpose(core.reshape(v, [B, T, self.n_head, C // self.n_head]), axes=[1, 2])

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = core.concatenate([past_key, k], axis=-2)
            v = core.concatenate([past_value, v], axis=-2)

        FULL_T = k.shape[-2]
        if use_cache is True:
            present = (k, v)
        else:
            present = None

        # causal self-attention (see drive for notes)
        # skip because we don't have flash
        # not sure if k.size is the right thing to put here
        att = (q @ core.transpose(k, [-2, -1])) * (1.0 / math.sqrt(k.size(-1)))
        att = 

        




