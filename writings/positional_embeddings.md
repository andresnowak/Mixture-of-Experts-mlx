## Absolute Embeddings
- This is the one where we just declare a simple matrix PE of size (max_sequence_length, embedding_dimension) and use this to add to our token embeddings the positional information
- This method makes it so that the model can learn arbitrary patterns of positional encoding tuned to the data (so it can learn a sinusoidal embedding, a repeating cycle or something else)
- But the problem with this method is that it can't generalize to longer sequences than the ones it was trained on

## Sinusoidal Embeddings
- For sinusoidal embeddings here we create a matrix PE of size (max_sequence_length, embedding_dimension) using the formula of $PE(pos, 2i) = sin(pos / (10000^{2i/d}))$ and $PE(pos, 2i + 1) = cos(pos / (10000))$ where $i$ is the index in the embedding_dimension (so cos is used in the odd indeces and sin in the even ones)
- This function is injective because if we see it as only one channel of the embedding dimension we need to have $|pos_1 - pos_2| = 2 \pi k \cdot 10000^{2_i / d}$ and majority of the times this is not possible because |pos_1 - pos_2| has to be an integer not a real number
  - But if we see it as the whole row vector for that pos (so the whole embedding dimension) it is possible to have a repeating value but only if the sequence length is very large (and we basically won't have that)
- This embeddings are called relative because $PE(pos) = PE(pos + k)$, the model only needs to learn a linear mapping function M that converts $M(K)PE(pos) = PE(pos + k)$
  - Remember the rules for $sin(a + b) = sin(a)cos(b) + cos(a)sin(b)$ and $cos(a + b) = cos(a)cos(b) - sin(a)sin(b)$, thanks to this it is possible to learn a linear function
  - So the model basically learns attention biases for this M(K)
- The advantages of sinusoidal embeddings is that the have the four properties of; periodicity, linearity, scale invariance and injective function.
  - So sinusoidal embedddings have the advantage that they can extend to bigger sequence lengths than the ones where the model was traineed on (because it only needed to learn relative distances, like if we trained with sequence length of 256 it learned relative distances of $0 \geq k \leq 255$), but performance still drops because the attention masking only learned for sequences of length equal to 256
- The disadvantages is that the model can't adapt the embedding space to the task at hand compared to the absolute ones, and when the size of your generation doesn't change (like it will always be the sequence length or smaller) absolute embeddings can outperform sinusoidal ones

## RoPE
[This can be a better explanation of RoPE](https://medium.com/@mlshark/rope-a-detailed-guide-to-rotary-position-embedding-in-modern-llms-fde71785f152#bypass) and this [one](https://blog.eleuther.ai/rotary-embeddings/)

The idea is that if we try to encode absolute positions in the token embeddings, we have
the problem that the dot product (and therefore attention) doesn't preserve this information:
the positional part just mixes with content and you lose a clear signal of distance. So
if we bake absolute positions into the embeddings, attention can’t tell how far apart two
tokens really are. We want scores that depend on $n-m$, not on $m$ or $n$ alone.

RoPE encodes position as a pure rotation in $\mathbb{C}^{d/2}$. Instead of working in
$\mathbb{R}^d$, we view each consecutive pair of real components as one complex number:
$$
q^{\mathbb{C}} = \bigl(q_1 + i\,q_2,\;q_3 + i\,q_4,\;\dots,\;q_{d-1} + i\,q_d\bigr)
\in \mathbb{C}^{d/2},
$$
and similarly for $k^{\mathbb{C}}$.

First recall from polar coordinates that any complex can be written as
$$
z = r\,e^{i\theta},\qquad e^{i\theta} = \cos\theta + i\,\sin\theta,
$$
where $r=|z|$ is the vector norm. Because $|e^{i\theta}|=1$, multiplying by $e^{i\theta}$
rotates $z$ without changing its length. Concretely, for $z = x + i y$:

$$(z)\,e^{i\theta} = (x + i\,y)(\cos\theta + i\,\sin\theta) = (x\cos\theta - y\sin\theta) + i\,(x\sin\theta + y\cos\theta)$$

Next assign each token position $m$ a phase $\phi_m = m\,\omega$. We rotate each 2D
subvector of the query by $e^{i\phi_m}$ and each subvector of the key by $e^{i\phi_n}$.

To see how the real dot product arises, take $z = x + i\,y$ and $w = u + i\,v$. The real
2D dot product $xu + yv$ is recovered by
$$
\Re\bigl[z\,\overline{w}\bigr]
= \Re\bigl[(x + i\,y)(u - i\,v)\bigr]
= x\,u + y\,v.
$$
In polar form $z = r_z e^{i\theta_z}$, $w = r_w e^{i\theta_w}$, we have
$$
z\,\overline{w} = r_z\,r_w\,e^{i(\theta_z - \theta_w)},
\quad
\Re[\cdot] = r_z\,r_w\cos(\theta_z - \theta_w).
$$
After rotating by $\phi_m,\phi_n$, the score in each complex plane is
$$
\Re\Bigl[(q_i^{\mathbb{C}}\,e^{i\phi_m})\,
         \overline{(k_i^{\mathbb{C}}\,e^{i\phi_n})}\Bigr]
= \|q_i\|\|k_i\|\cos(\phi_m - \phi_n).
$$
Since $\phi_m - \phi_n = (m-n)\,\omega$, this depends *only* on $n-m$.

Equivalently, each 2D pair undergoes the real rotation
$$
R(\theta) =
\begin{pmatrix}
  \cos\theta & -\sin\theta\\
  \sin\theta &  \cos\theta
\end{pmatrix}.
$$
Define
$$
f_q(x_m,m) = R(m\omega)\,q,\quad
f_k(x_n,n) = R(n\omega)\,k,
$$
with $q = W_q\,x_m$, $k = W_k\,x_n$. Then
$$
\langle f_q(x_m,m), f_k(x_n,n)\rangle
= q^\top R((n-m)\omega)\,k$$