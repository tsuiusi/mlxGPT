# mlxGPT: Rewrite of Karpathy's NanoGPT in mlx

---
## Devlog
### 9th Feb
* I don't need infinity for the mask, a sufficiently negative number will suffice

---
## Notes of the rewrite
Components of NanoGPT:
* LayerNorm
* Causal Self-Attention
* Multi-Layer Perceptron (MLP)
* GPT

> *gradatim ferociter*

### LayerNorm
* https://arxiv.org/pdf/1607.06450.pdf
* Counters the highly-correlated nature of batch norm
* Batch norm has to estimate because it’s impractical to go through all the weights (Eq. 2)
* µ and σ are calculated using the empirical samples from the current mini-batch, which constraints the size of the batch and is hard to apply to RNNs. 

Specifically, for the $i^{th}$ summed input in the $l^{th}$ layer, the batch normalization method rescales the summed inputs according to their variances under the distribution of the data:

![batchnorm](/images/batchnorm.png)

Where $a^{-l}\_i$ is normalized summed inputs to the $i^{th}$ hidden unit in the $l^{th}$ layer and $g_i$ is a gain parameter scaling the normalized activation before the non-linear activation function.
}
> $a^l$: the vector representation of the summed inputs to the neurons in that layer

> $g_i$: gain parameter, used to scale the weights, helps control the variance of the outputs of neurons

> $σ\_i^\'$: the standard deviation of activations $a\_i^\'$ over the batch of data

> $µ\_i^\'$: mean of activations of $a\_i^\'$ over the batch of data


**Layer normalization**
* Reduces highly correlated changes in the summed inputs to the next layer (especially ReLU)
* "Covariate shift" problem can be reduced by fixing the mean and variance of summed input (µ and σ) 
* The layer normalization statistics over all hidden units in the layer are computed with the following equations:

![layernorm](/images/layernorm.png)

Where H denotes the no. hidden units in the layer. Unlike BatchNorm, LayerNorm does not impose any constraint on the size of a mini-batch and can be used in the pure online regime (?) with batch size 1.

Note that the normalization terms only depend on the summed inputs to a layer in the *current time step*. It also has *only one set* of gain and bias parameters shared over all time steps.

**In RNN**

The summed inputs in the recurrent layer are computed from the current input $x^t$ and previous vector of hidden states $h^{t-1}$, which are computed as $a^t$ = $W_{hh} h^{t-1} + W_{xh} x^t$. 

The LayerNorm-ed recurrent layer recenters and rescales the activations using extra terms:
![extraterms](/images/extraterms.png)

Where $W_{hh}$ is the recurrent hidden to hidden weights and $W_{xh}$ are the bottom up input to hidden weights.
> The O. thing is the elem-wise multiplication between 2 vectors
> b & g are the bias and gain params, same dimension as $h^t$

> In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed inputs to a layer, which results in much more stable hidden-to-hidden dynamics.









