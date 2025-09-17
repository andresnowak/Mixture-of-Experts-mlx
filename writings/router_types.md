## Router types
- **The common top-k experts for each token (from *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*)** (*This implementation is still difficult for me to understand*)
  - $$G(x) =\mathrm{Softmax}\,\bigl(\mathrm{KeepTopK}\bigl(H(x),\,k\bigr)\bigr)$$
  - $$H(x)_i =(x\,W_g)_i + \mathcal{N}(0,1) \times \mathrm{Softplus}\bigl((x\,W_{\text{noise}})_i\bigr)$$
  - $$\mathrm{KeepTopK}(v,k)_i =
    \begin{cases}
    v_i,&\text{if }v_i\text{ is among the top-}k\text{ entries of }v,\\
    -\infty,&\text{otherwise.}
    \end{cases}
  $$
  - Here the idea is that we want to select $K$ experts for each token. Also here we can see that the softmax will be done instead on only the $K$ chosen using the logits instead of the one in DeepSeek-MoE where we choose the $K$ from the softmax result instead (and here the probabilities are not normalized (because after the selection they will not sum to 1))
  - In this paper they also use the noise and softplus (that will control the amount of noise per component)
    - The idea of the Noise is to have load balance, because the amount of tokens an expert receives is something discrete we can't have backpropagation
      - So we have first that we wil model each variable as a Guassian random variable where $(x\,W_g)_i + \sigma \cdot \mathcal{N}(0,1)$ and we are asking what is the probability that my logits are bigger than the biggest logit from all the other experts
      - And from this we can basically do another type of Load balance
    - But in the end instead the the noise factor has stop being used because it seems it was more costly and difficult to tune the **CV** factor,
      - **And instead we use the Switch balance loss or other regularizers, and with this it is not necessary to use the Noise to do the balancing loss**
        - $$G(x)  = \mathrm{Softmax}\,\bigl(\mathrm{KeepTopK}\bigl(H(x),\,k\bigr)\bigr)$$
        - $$H(x)_i  = (x\,W_g)_i$$
        - $$\mathrm{KeepTopK}(v,k)_i =
          \begin{cases}
          v_i,&\text{if }v_i\text{ is among the top-}k\text{ entries of }v,\\
          -\infty,&\text{otherwise.}
          \end{cases}
        $$
        - Or we do the TopK after the softmax instead

- **Expert Choice Routing (from *Mixture-of-Experts with Expert Choice Routing*)**
<img src="./images/expert-choice-routing.png" width="600px"></img>
  - For this version now the idea is that the amount experts used for a token can be variable (instead of having a forced $K$ amount of experts), and here instead each expert can only use $K$ amount of tokens
  - So basically this version tries to fix the problems of:
    - Load imbalance: Because it is possible for some experts to be under-utilized during training and a sub-optimal strategy can produce redundant experts and/or experts that are not sufficiently specialized
      - But at least I think *DeepSeek-MoE* fixes this somewhat using the idea of shared experts and just having more experts to use in general
  - And supposedly this method allows for perfect load balancing despite its simplicity, using variable number of experts for each token, and achieves substantial gains in training efficiency and downstream task performance
  - Formula:
    - Let $X \in \mathbb{R}^{n \times d}$ be the token representations. The routing produces
		$I, G, P$ where:
		- $I \in \{1,\dots,n\}^{e \times k}$ is an index matrix; $I[i,j]$ is the index of the $j$-th selected token for expert $i$.
		- $G \in \mathbb{R}^{e \times k}$ are the gating weights for the selected tokens.
		- $P \in \{0,1\}^{e \times k \times n}$ is the one-hot version of $I$ used to gather/scatter tokens.

		The gating function is:
		$$
		S = \mathrm{Softmax}(X W_g), \qquad S \in \mathbb{R}^{n \times e}, W_g \in \mathbb{R}^{d \times e}
		$$

		$$
		G,\, I = \mathrm{TopK}(S^\top,\, k), \qquad P = \mathrm{OneHot}(I)
		$$

		Inputs to each expert’s FFN are gathered by $P$:
		$$
		X_{\mathrm{in}} = P \cdot X, \qquad X_{\mathrm{in}} \in \mathbb{R}^{e \times k \times d}
		$$

		Per-expert FFN (biases omitted), with $W_1[i], W_2[i] \in \mathbb{R}^{d \times d'}$:
		$$
		\forall i:\quad X_e[i] = \mathrm{GeLU}\!\big(X_{\mathrm{in}}[i]\, W_1[i]\big)\, W_2[i]^\top
		$$

		Final MoE layer output (combining permutation $P$ and gates $G$):
    $$
    X_{\mathrm{out}}[l, d] = \sum_{i=1}^{e} \sum_{j=1}^{k} P[i, j, l] \, G[i, j] \, X_e[i, j, d],
    \qquad X_{\mathrm{out}} \in \mathbb{R}^{n \times d}.
    $$
  - They set the number of tokens each expert can have as $top\_k = \frac{n \cdot c}{e}$
  	- Where $n$ is the total number of tokens in the input batch (batch_size x sequence length), $c$ is the capacity factor (it denotes on average how many experts are utilized by a token), and $e$ is the number of experts
  	- **But because of this capacity factor, it is possible for some tokens to be dropped, and that's why it seems with this version it helps to have the MoE instead on all layers to only have it for example in every other layer** (or at least thats their theory)
    	- And even this problem can happen in the other routing versions because sometimes capacity factors are also used and in the implementation of **OpenMoE** they showed that in pretraining it work correctly but then going to finetuning (because it is a little bit out-of-distribution) the amount of tokens drop would get high. And their idea to fix this was to use part of the finetuning data in the pretraining (Not for alignment but to help with load balancing)
	- **Limitations**
  	- Lastly this implementation has the problem that it might not work in auto-regressive generation, because the method takes into account past and future tokens to do the top-k selection
    	- During teacher forcing it does have access to the past and future information, so it is possible to train the Decoder LLM with this, but then when generating at each step the amount of tokens is different at each step (**and the capacity factor is based on the total amount of tokens when doing the teacher forcing**), so the behaviour of the model could break the load balance and even the performance of the model maybe
  - **Questions**:
    - For this routing method I don't understand why is it that they apply the softmax over the expert dimension and no the token dimension
