import math
from dataclasses import dataclass

import mlx
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten

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
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias-config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
       
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.transformer = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # skip weight initialization because we're using mlx :)

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
		params are actually used as weights in the final layer, so we include them.
		"""
        n_params = sum(p.size for _, p in [tree_flatten(self.transformer.parameters(), tree_flatten(self.ln_f.parameters()), tree_flatten(self.lm_head.parameters()))])

        if not non_embedding: 
            n_params += tree_flatten(self.wpe).size

        return n_params

    def __call__(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = mx.arange(0, t, 1, dtype=idx.dtype) # shape (t)

        # forward the GPT model itself
        tok_embd = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_embd = self.wpe(idx) # position embeddings of shape (t, n_embd)
        x = self.drop(tok_emb +  pos_emb)
        for block in self.transformer:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        else:
             # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
		# model surgery to decrease the block size if necessary
		# e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
		# but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.wpe = self.wpe.parameters()["weight"][:block_size]
        for block in self.transformers:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[: ,: , :block_size, :block_size]
                
        
    def generate(self, idx, max_new_tokens=512, temp=1.0, top_k=None):
        for _

