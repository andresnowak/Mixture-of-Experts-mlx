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
- The advantages of sinusoidal embeddings is that the have the four properties of; periodicity, linearity, scale invariance and injective function.
  - So sinusoidal embedddings have the advantage that they can extend to bigger sequence lengths than the ones where the model was traineed on (because it only needed to learn relative distances, like if we trained with sequence length of 256 it learned relative distances of $0 \geq k \leq 255$), but performance still drops because the attention masking onlyh learned for sequences of length equal to 256
- The disadvantages is that the model can't adapt the embedding space to the task at hand compared to the absolute ones, and when the size of your generation doesn't change (like it will always be the sequence length or smaller) absolute embeddings can outperform sinusoidal ones
