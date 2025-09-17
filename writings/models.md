## DeepSeek-MoE
<img src="./images/deepseek-moe.png" width="600px"></img>

The two important contributions of DeepSeek-MoE are
 - Finely segmenting the exeprts
  - The idea is that if you have $N$ experts instead we will finely segment this experts more to have $mN$ experts, and the same for the routed experts where now we will have $mK$ routed experts
  - This idea helps with the problem of **Knowledge hybridity** (where an expert will have to mix a lot of differnt knowledge into its parameters instead of bieng more specialized), and consequently encourages sharper specialization
- Shared expert isolation
  - We isolate $K_s$ experts as shared ones (this experts are always active)
  - The idea here is that this models will capture general knowledge so as to reduce knowledge redundancy between the other experts (e.g. there is one expert that learns biology and another Laws, but in the end both have to learn about language, so instead we can have a shared expert that learns about language instead).
  - In the end the implementation looks like this:
  	- $h_t^l = \sum_{i=1}^{K_s} \mathrm{FFN}_i\!\left(u_t^l\right) + \sum_{i=K_s+1}^{mN} g_{i,t}\,\mathrm{FFN}_i\!\left(u_t^l\right) + u_t^l$
    - $$ g_{i,t} =
      \begin{cases}
      s_{i,t}, & s_{i,t} \in \mathrm{Topk}\!\left(\{\, s_{j,t} \mid K_s + 1 \le j \le mN \,\},\, mK - K_s\right) \\
      0, & \text{otherwise}
      \end{cases}
      $$
    - $s_{i,t} = \mathrm{Softmax}_i\!\left( (u_t^l)^\top e_i^l \right)$
    - Where:
      - $e^l_i$ ($i$ is the expert and $l$ is the layer) is the learned weight embeddings of the experts where $e \in R^{\text{hidden dim } \cdot \text{ number of experts}}$
      - $u$ is the result of the final projection of the Attention layer, where $u \in R^{\text{batch size} \cdot \text{seq len} \cdot \text{hidden dim}}$
      - $s$ are basically what we call the affinities that each token has to an expert, based on this we choose the experts that have the biggest affinity to a specific token and always our shared expert
      - $s$ are the end result affinities where the experts that where not chosen get a $g = 0$, so in the end they don't contribute anything to $h$, adn this is why MoE are sparse
