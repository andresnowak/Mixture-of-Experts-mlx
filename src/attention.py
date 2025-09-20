import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mlx_extension import sink_softmax
from .positional_embeddings import RoPE

class MultiHeadAttention(nn.Module):
    # Multi-head attention, implementation from https://arxiv.org/abs/1706.03762
    def __init__(self, max_seq_len: int, emb_dim: int, num_heads: int, bias: bool=True, use_rope: bool=False):
        # Dimension of model (emb_dim) is equal to num_heads * d_k, d_k = emb_dim / num_heads
        # emb_dim has to be divisble by num_heads

        # Q and K use the same d_k and we also say that d_v = d_k

        super().__init__()

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        d_k = emb_dim // num_heads
        self.scaling: float = 1 / np.sqrt(d_k)

        self.rope = None
        if use_rope:
            self.rope = RoPE(max_seq_len, emb_dim)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=bias)

    def _split_heads(self, z: mx.array):
        batch, seq_len, _ = z.shape

        return z.reshape(
            batch, seq_len, self.num_heads, self.emb_dim // self.num_heads
        ).transpose(0, 2, 1, 3)  # (B, num_heads, seq_len, d_k)

    def _apply_mask(self, qk: mx.array, attn_mask: mx.array | None) -> mx.array:
        """Apply attention mask to QK scores."""
        if attn_mask is not None:
            mask = mx.where(attn_mask, 0, -float("inf"))
            qk = qk + mask.reshape(1, 1, *attn_mask.shape)
        return qk

    def _compute_attention(self, Q: mx.array, K: mx.array, V: mx.array, attn_mask: mx.array | None) -> mx.array:
        """Compute attention scores and apply to values. Can be overridden by subclasses."""
        score = self._apply_mask(Q @ K.transpose(0, 1, 3, 2), attn_mask) * self.scaling
        attention = mx.softmax(score, axis=-1) @ V
        return attention

    def _post_attention_processing(self, attention: mx.array, x: mx.array) -> mx.array:
        """Post-process attention output. Can be overridden by subclasses."""
        batch, seq_len, _ = x.shape
        multi_head = attention.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.emb_dim)
        return self.W_o(multi_head)

    def __call__(self, x: mx.array, attn_mask: mx.array | None = None) -> mx.array:
        """
        Multi-head scaled-dot-product attention.

        Parameters
        ----------
        x : array
            Input sequence of shape (batch, seq_len, embed_dim).
        attn_mask : array or None, optional
            Boolean mask of shape (batch, seq_len, seq_len).
            Positions that are ``False`` will be masked out (set to -inf)
            before the softmax.  If ``None``, no masking is applied.

        Returns
        -------
        array
            Output tensor of shape  (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)  # (B, seq_len, emb_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attention = self._compute_attention(Q, K, V, attn_mask)
        return self._post_attention_processing(attention, x)


class GatedAttention(MultiHeadAttention):
    # Gated Attention, implementation from https://arxiv.org/abs/2505.06708
    def __init__(self, max_seq_len: int, emb_dim: int, num_heads: int, bias: bool=True, use_rope: bool=False):
        super().__init__(max_seq_len, emb_dim, num_heads, bias, use_rope)
        self.W_gates = nn.Linear(emb_dim, num_heads, bias=False) # head specific gating not elementwise

    def _post_attention_processing(self, attention: mx.array, x: mx.array) -> mx.array:
        """Apply gating to attention output before final projection."""
        batch, seq_len, _ = x.shape

        # Transpose to (batch, seq_len, num_heads, d_k)
        attention = attention.transpose(0, 2, 1, 3)

        # Apply the gating
        gate = mx.expand_dims(mx.sigmoid(self.W_gates(x)), axis=-1) # (batch, seq_len, h, 1)
        attention_gating = attention * gate # (batch, seq_len, h, d_k)

        # Reshape and apply final projection
        attention_gating = attention_gating.reshape(batch, seq_len, self.emb_dim)
        return self.W_o(attention_gating)


class BiasedAttention(MultiHeadAttention):
    # Biased Attention from GPT-OSS, implementation from https://arxiv.org/pdf/2508.10925
    def __init__(self, max_seq_len: int, emb_dim: int, num_heads: int, bias: bool=True, use_rope: bool=False):
        super().__init__(max_seq_len, emb_dim, num_heads, bias, use_rope)
        self.sinks = mx.zeros(self.num_heads, 1) # For the biased denominator softmax

    def _compute_attention(self, Q: mx.array, K: mx.array, V: mx.array, attn_mask: mx.array | None) -> mx.array:
        """Use sink softmax instead of regular softmax."""
        score = self._apply_mask(Q @ K.transpose(0, 1, 3, 2), attn_mask) * self.scaling
        attention = sink_softmax(score, self.sinks) @ V
        return attention
