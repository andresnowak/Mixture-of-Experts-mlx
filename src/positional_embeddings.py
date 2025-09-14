import mlx.core as mx

def absolute_embeddings(max_length: int, emb_dim: int) -> mx.array:
    return mx.random.normal((max_length, emb_dim))


def sinusoidal_embeddings(max_length: int, emb_dim: int) -> mx.array:
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
