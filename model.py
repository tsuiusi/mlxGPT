import math
from dataclasses import dataclass

import mlx
import mlx.nn as nn
import mlx.core as mx

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        # I don't think I need to initialize the weights and biases because lazy eval
        self.weights = mx.ones((ndim, ))
        self.bias = mx.ones((ndim, )) if bias else None

    def forward(self, x):
        return nn.LayerNorm(x, self.weights, dims=ndim, eps=1e-5) 

class LayerNorm2(nn.Module):
    # Affine is a type of layer where each input is connected to each output by a learnable weight (in other words, fully connected)
    # eps is epsilon, the tolerance for how close the solution needs to be to 0 before it's considered (?)
    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = False):
        super().__init__()
        if affine:
            self.bias = bias
            if bias:
                self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def __call__(self, x):
        # in pytorch the mean and standard deviation (standard deviation = √var(x)) is calculated over the last D dimensions and D is the dimension of normalized_shape. 
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        # first part of the pytorch equation, input - E(x)/√var + eps
        # E(x) is mean
        x = (x - mean) / mx.rsqrt(var + self.eps)
        if self.bias:
            return (self.weight * x + self.bias) if "weight" in self else x
        else:
            return (self.weight * x) if "weight" in self else x



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
        self.flash = True

    def __call__(self, x, cache=None):
        B, T, C = x.shape
        
        # query, key, value for all heads in batch and move head forward to be the batch dim
        q, k, v = mx.split(self.c_attn(x), 3, axis=2) # Splits into 3x768 instead of 768x3
        k = mx.transpose(mx.reshape(k, [B, T, self.n_head, C // self.n_head]), axes=[0, 2, 1, 3])
        q = mx.transpose(mx.reshape(q, [B, T, self.n_head, C // self.n_head]), axes=[0, 2, 1, 3])
        v = mx.transpose(mx.reshape(v, [B, T, self.n_head, C // self.n_head]), axes=[0, 2, 1, 3])

        if cache is not None:
            past_key = cache[0]
            past_value = cache[1]
            k = mx.concatenate([past_key, k], axis=-2)
            v = mx.concatenate([past_value, v], axis=-2)

        # causal self-attention (see drive for notes)
        # not sure if k.size is the right thing to put here
        # att = (q @ mx.transpose(k, [-2, -1])) * (1.0 / math.sqrt(k.shape(-1)))
        # mask = mx.reshape(mask, (1, 1, T, T))
        # att = mx.where(mask[:, :, :T, :T] == 0, att , float('-inf'))
        # att = mx.softmax(att, axis=-1)
        # att = self.attn_dropout(att)

        # Because we have flash attention we use flash attention
        y  = mx.fast.scaled_dot_product_attention(q=q, k=k, v=v, scale=(1.0 / math.sqrt(q.shape[-1])), mask=None) 
        y = y.transpose(0, 2, 1, 3).reshape(B,T,C) # reassemble all the head output side by side; this line is inconsistent with the transpose because i just copied karpathy
        # he used contiguous here which returns a tensor in the format memory likes, but no such thing in mlx so 0213 it is

        # resid_dropout randomly dropouts residual connections (connections to prev layers) 
        # c_proj is a linear transformation (projection) applied to the output, which either changes the dimension or prepares layers or something like that.
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias) #4*config.n_embd because choice of design, bias is bool
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) # probability
    
    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm2(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm2(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x, cache=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

@dataclass # defines that this is a dataclass and not a normal class
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12    
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster	
    

class GPT(nn.Module):
    """
    What the GPT contains:
    1. init: initializes everything, all the hyperparams, etc
    2. generate
    3. optimizer
    4. forward
    5. estimating model flop utilizaiton (mfu) (kinda unnecessary for this application so far)
    
    Karpathy also included loading pretrained models but I don't feel like doing that rn. Implementing this needs crop_block_size as a prereq.
    """


    """
    To clarify:
    1. Layernorm takes the input tensor, dimensions, and whatever hyperparams that come preconfigured
    2. CSA takes the input tensor, mask, and cache, which is preset to None
    3. MLP takes input tensor only
    4. Block takes input tensor, mask, and cache, which is again preset to None. I cba to implement it rn
    """

    def __init__(self, config):
        """
        Initializes a GPT model.

        config configures the hyperparameters of the GPT model
        embedding (nn.Embedding) creates the embedding layer for input tokens
        transformer (List[Block]) is a list of transformer blocks
        lm_head (nn.Linear) is the linear layer for output projection
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # Word token embeddings, converts each token in the input sequence into a fixed-size vector.
        self.wpe = nn.Embedding(config.block_size, config.n_embd) # Word position embeddings, encodes the position of each token in the sequence 
        # input representation = wte + wpe
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm2(config.n_embd, bias=config.bias)
        self.transformer = [Block(config) for _ in range(config.n_layer)] # creates n no. blocks 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # takes the no. embeddings as input, outputs tensor of size vocab_size to be one-hotted 

    def sample_next_token(self, x, temp):
        logits = mx.expand_dims(x[:, -1], axis=0) @ self.wte.weight.T
        y = logits[:, -1, :]
        y = mx.random.categorical(y * (1/temp))
        return y

    def __call__(self, x, targets=None):
        b, t = x.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=x.dtype)

        # forward the GPT model 
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)

        x = self.drop(tok_emb + pos_emb)

        for block in self.transformer:
            x = block(x)

        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        # if targets is not None:
            # logits = self.lm_head(x)
            # there might be something wrong here i'll have to experiment on this. i don't fully get mlx.core.reshape
            # loss = nn.losses.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        # else:
            # logits = self.lm_head(x)
            # loss = None
            
        return logits

    # let me think about this
    # What is the process of generation
    # Get the input, forward it to get the next token, get the logits at the final step and scale with temperature, (optionally) crop them, apply softmax to get probabilities, sample from the distribution, append to the current sequence, continue.
    def generate(self, idx, max_new_tokens=512, temp=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temp
            
            # Optionally crop the logits to only the top_k options
            if top_k is not None:
                v, _ = topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities
            probs = mx.softmax(logits)
            # Sample from distribution
            idx_next = mx.random.categorical(probs, 1)
            # Append to current sequence
            idx = mx.concatenate([idx, mx.expand_dims(idx_next, axis=0)], axis=1)

        return idx

    

def topk(x, k):
    flatten = mx.reshape(x, (-1,))
    sorted_idx = mx.argsort(flatten)
    sorted_idx = mx.take(sorted_idx, mx.arange(sorted_idx.size -1, -1, -1))

    topk_indices = mx.take(sorted_idx, mx.arange(0, k))
    topk_values = mx.take(flatten, topk_indices)
    return mx.expand_dims(topk_values, axis=0)
