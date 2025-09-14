## Gated Attention for LLM's - Non-Linearity, Sparsity, and Attention-Sink-Free [paper](https://arxiv.org/abs/2505.06708)
 
In this paper they investigate and do experiment with different gating mechanism in the standard softmax attention.

First lets formalize the gating mechanism:
 $\hat{Y} = g(Y, X, W_\theta, \sigma) = Y \odot \sigma(XW_\theta)$ 
- Where:
	- $Y$ is the input to be modulated, 
	- $X$ is another input used to compute the gating scores
	- And $W_\theta$ refers to the learnable parameters for the Gate

Here $\sigma(XW_\theta)$ acts as a dynamic filter, controlling the information flow from $Y$ by selecting or erasing its features



In the end they found two methods that give the best results:
- Value gating
  - sfd
- And Scaled Dot Production Attention gating

