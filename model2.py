import math

import mlx
import mlx.nn as nn
import mlx.core as mx

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.ndim = ndim
        self.bias = bias
    
    def __call__(self, x):
        return nn.LayerNorm(x, dims=self.ndims, bias=self.bias)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch 
        sely.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
       	# regularization
		self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr
        # no need nanoGPT check here because mlx supports flashattention (≧▽≦)

    def __call__(self, x):
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, axis=2)
        k = k.reshape([B, T, self.num_heads, C//self.num_heads]).transpose(axes=[0, 2, 1, 3]) # (B, nh, T, hs)
        q = q.reshape([B, T, self.num_heads, C//self.num_heads]).transpose(axes=[0, 2, 1, 3]) # (B, nh, T, hs)
        v = v.reshape([B, T, self.num_heads, C//self.num_heads]).transpose(axes=[0, 2, 1, 3]) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=(1.0/math.sqrt(q.shape[-1])), mask=None)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

