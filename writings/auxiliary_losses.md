## Auxiliary losses
Automated learned routing strategies may encounter load imbalance, where
  - The model always select only a few experts (*routing collapse*)
  - And if the experts are distributed across multiple devices, then we can create computation bottlenecks basically

For this we use:
- **Expert level balance loss (from *DeepSeek-MoE*)**
  - Here the shared experts don't contribute to the loss as they are always chosen
  - $L_\text{ExpBal} = \alpha_1 \sum^{\hat{N}}_{i=1}f_i P_i$
  - $f_i = \frac{\hat{N}}{\hat{K}T} \sum^T_{t=1} \mathrm{1}(\text{Token t selects expert i})$
  - $P_i = \frac{1}{T} \sum^T_{t=1} s_{i,t}$
  - Where:
    - $\alpha_1$ is a hyperparameter called "expert-level balance factor", $\hat{N}$ is equal to ($mN - K_s$) and $\hat{K} = (mK-K_s)$ (so basically the experts that are not shared experts), $\mathrm{1}(\cdot)$ is the indicator function (where 1 if true and 0 if not) and T is the total number of tokens so $\text{batch size} \cdot \text{sequence length}$ (we basically balance across the whole batch)
  - This loss forces the way the experts are chosen to be uniform. Because, if it decides to always choose the same experts for every token ($s_{i,t}$ always small for everyone else), it's value will get bigger (because from the load balance loss we can see how our average $s_i$ (that is $P_i$) over all the tokens will get multiplied by $f_i = \hat{N}/\hat{K}$ (because the sum of selected tokens will be equal to $T$)), so it is better to balance the values because like this $f_i$ and $P_i$ values will be smaller.
  - And here there is no problem for auto-regression as this auxiliary loss is only used during the training (so using total tokens in the whole batch for this there is no problem).
    - The thing is as in the *OpenMoE* paper showed, this routing function is learned during the pre-training but at fine-tuning the distribution of tokens can be Out-of-distribution (That it is a very different type of corpus to learn in how it is structured), and here the routing rule in the end can collapse a little where now some experts will receive a bigger amount of tokens than others (now we don't have a good load balance)

- **Load balancing loss (from *Switch transformer*)**
  - loss = $\alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$ (4)
    - where $f_i$ is the fraction of tokens dispatched to expert $i$, $$f_i = \frac{1}{T} \sum_{x \in B} \mathbb{1} \{\mathop{\mathrm{argmax}} p(x) = i \}$$ (5)
    - and $P_i$ is the average of the router probability allocated for expert $i$ over all the tokens, $$P_i = \frac{1}{T} \sum_{x \in B} p_i(x)$$
    - Here again if the router keeps favoring the same experts, they will get $f_i \approx 1$ and the sum of $P_i$ will approximate 1 only for the chosen experts and the others it will go to 0. And in the end the loss will be $\alpha \cdot N$
      - And if instead it is uniformly across all experts we will have $f_i = 1/N$ and $P_i = 1/N$ so in the end we have $\alpha \cdot N \cdot N \cdot 1/N^2 = \alpha$
- **Noisy Top-K load estimator and load-balance loss (from *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*)** ((This implementation is still difficult for me to understand))
  - $$
  P(x, i)
  = \Pr\Bigl((x \cdot W_g)_i + \mathcal{N}(0,1)\,\times\,\mathrm{Softplus}\bigl((x \cdot W_{\mathrm{noise}})_i\bigr) > \mathrm{kth\_excluding}\bigl(H(x),\,k,\,i\bigr)\Bigr)$$
  - where `kth_excluding(v, k, i)` returns the $k$-th largest entry of $v$ *excluding* the $i$-th. (So basically here they are doing a z-score to then grab from the the table of the Normal gaussian CDF?)
  - $$ P(x, i = \Phi\!\Bigl(
    \frac{(x \cdot W_g)_i - \mathrm{kth\_excluding}(H(x),k,i)}
         {\mathrm{Softplus}\bigl((x \cdot W_{\mathrm{noise}})_i\bigr)}
  \Bigr)$$ with $\Phi$ being the CDF of the standard normal.

  - $$\mathrm{Load}(X)_i = \sum_{x \in X} P(x, i)$$
  - $$\mathcal{L}_{\mathrm{load}}(X) = w_{\mathrm{load}}\bigl(\mathrm{CV}\bigl(\mathrm{Load}(X)\bigr)\bigr)^2$$

    - Here **CV** is the *coefficient of variation* of the load vector $v$ (This part of **CV** I don't understand it)
  - But basically models use instead the other load balance losses and remove the Noise from the routing as it seems it was a very difficult to tune the **CV** and in the end it seems more simple regularizers or hard capacity factors give better results
