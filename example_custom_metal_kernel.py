import mlx.core as mx

source = """
    uint elem = thread_position_in_grid.x;
    T tmp = inp[elem];
    out[elem] = tmp*tmp;
"""

kernel = mx.fast.metal_kernel(
    name="myexp",
    input_names=["inp"],
    output_names=["out"],
    source=source,
)

def exp_elementwise(a: mx.array):
    outputs = kernel(
        inputs=[a],
        template=[("T", mx.float32)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )
    return outputs[0]

a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
b = exp_elementwise(a)
assert mx.allclose(b, a**2)

print(a)
print(b)
