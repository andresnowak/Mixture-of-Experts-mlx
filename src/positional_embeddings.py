import mlx.core as mx
import mlx.nn as nn

def absolute_embeddings(max_length: int, emb_dim: int) -> mx.array:
    return mx.random.normal((max_length, emb_dim))


def sinusoidal_embeddings(max_length: int, emb_dim: int) -> mx.array:
    assert emb_dim % 2 == 0
    positions = mx.expand_dims(mx.arange(start=0, stop=max_length, step=1), axis=-1)

    dimensions = mx.arange(
        start=0, stop=emb_dim, step=2
    )  # remember we have 2i and 2i+1, meaning even part is sin and odd is cos

    embeddings = mx.zeros((max_length, emb_dim))

    frequencies = mx.exp(-dimensions * (mx.log(10_000) / emb_dim))

    # sin to even positions and cos to odd positions

    embeddings[:, 0::2] = mx.sin(positions * frequencies) # (max_length, 1) * (emb_dim / 2) = (max_length, emb_dim / 2)
    embeddings[:, 1::2] = mx.cos(positions * frequencies) # (max_length, 1) * (emb_dim / 2) = (max_length, emb_dim / 2)

    return mx.stop_gradient(embeddings)

class RoPE(nn.Module):
    def __init__(self, max_length: int, emb_dim: int):
        super().__init__()

        positions = mx.expand_dims(mx.arange(start=0, stop=max_length, step=1), axis=-1)

        dimensions = mx.arange(
            start=0, stop=emb_dim, step=2
        )  # remember we have 2i and 2i+1, meaning even part is sin and odd is cos

        frequencies = mx.exp(-dimensions * (mx.log(10_000) / emb_dim))
        # outer product => [max_seq_len, emb_dim/2], each position has its frequencies
        frequencies = mx.outer(positions, frequencies)

        embeddings = mx.stack([frequencies, frequencies], axis=-1).reshape(max_length, emb_dim) # [max_seq_len, emb_dim]

        self.cos = embeddings.cos()
        self.sin = embeddings.sin()

    def __call__(self, x: mx.array) -> mx.array:
        '''
        RopE: Apply before spliting heads

        Parameters
        ----------
        x : array
            Input sequence of shape (batch, seq_len, embed_dim).

        Returns
        -------
        array
            The result of applying our rotations to each consecutive pair of x
        '''
        batch, seq_len, emb_dim = x.shape

        # the padding is done from right to left (so self.cos will be [1, max_length, emb_dim])

        r_even = x[:, :, 0::2] * self.cos[:seq_len, 0::2] - x[:, :, 1::2] * self.sin[:seq_len, 1::2]
        r_odd = x[:, :, 1::2] * self.cos[:seq_len, 1::2] + x[:, :, 0::2] * self.sin[:seq_len, 0::2]

        out = mx.stack([r_even, r_odd], axis=-1).reshape(x.shape)

        return out
