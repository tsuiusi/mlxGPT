import mlx
import mlx.nn as nn
import mlx.core as core

import torch
import torch.nn as tnn
import torch.nn.functional as F

# Example tensor of scores (e.g., attention scores before applying softmax)
scores = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 1.0],
    [3.0, 4.0, 1.0, 2.0],
    [4.0, 1.0, 2.0, 3.0]
])

# Mask indicating valid (1) and invalid (-inf) positions
# For example, we want to mask out (ignore) the last two positions in each row
mask = torch.tensor([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
], dtype=torch.bool)

# Use masked_fill to set masked positions in 'scores' to -inf
masked_scores = scores.masked_fill(mask, float('-inf'))

# Apply softmax (usually done in attention mechanisms)
# Use dim=1 to apply softmax across each row
softmax_scores = F.softmax(masked_scores, dim=1)

maskedinf = scores
maskedinf[mask] = float('-inf')

scores = core.array([
     [1.0, 2.0, 3.0, 4.0],
     [2.0, 3.0, 4.0, 1.0],
     [3.0, 4.0, 1.0, 2.0],
     [4.0, 1.0, 2.0, 3.0]
])

mask = core.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
    ]).astype(core.bool_)
    
print(scores)
print(mask)

mask = core.where(mask, core.array(float('-inf')), core.array(0))
print(mask)
