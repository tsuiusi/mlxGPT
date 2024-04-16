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

WEIGHTS_PATH = "~/rtty/code/mlxGPT/124M/
