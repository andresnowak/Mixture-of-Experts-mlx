import mlx.core as mx


def multinomial(
    x: mx.array, num_samples: int = 1, replacement: bool = False
) -> mx.array:
    assert 1 <= x.ndim <= 2 and num_samples > 0, (
        f"{x.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    )
    assert replacement or num_samples == 1, (
        "no replacement only supports num_samples = 1"
    )

    weights = mx.expand_dims(x, 0) if x.ndim == 1 else x

    B, D = weights.shape

    cdf = (
        (cw := mx.cumsum(weights, axis=-1)) / cw[:, -1]
    )  # so as to make the values in the array sum to 1, and we also get the last value be equal to 1.0

    unif_samples = mx.random.uniform(shape=[B, num_samples])

    U_exp = mx.expand_dims(unif_samples, 2)  # (B, num_samples, 1)
    cdf_exp = mx.expand_dims(cdf, 1)  # (B, 1, D)
    mask = U_exp <= cdf_exp  # (B, num_samples, D)

    indices = mask.argmax(
        axis=2
    )  # (B, num_samples), so basically here we are grabbing the first value that gives true, because argmax iterates from first position to last and it grabs the first value it finds

    # 6) if the original x was 1-D, drop the batch dimension
    if x.ndim == 1:
        indices = indices.reshape((num_samples,))

    return indices  # (B, num_samples)


def one_hot(x: mx.array, num_classes: int):
    # assert x.dtype in [mx.uint8, mx.uint16, mx.uint32, mx.uint64, mx.int8, mx.int16, mx.int32, mx.int64, mx.unsignedinteger, mx.integer]

    orig_shape = x.shape

    eye_mat = mx.eye(num_classes)
    one_hot = eye_mat[x.reshape(-1)]  # (orig_shape.flatten(), num_classes)

    return one_hot.reshape(*orig_shape, num_classes)
