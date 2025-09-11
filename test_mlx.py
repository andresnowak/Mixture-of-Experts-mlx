import mlx.core as mx


a = mx.array([1, 0.9, 1, 1,])

print(mx.argpartition(a, 2)[:2])
