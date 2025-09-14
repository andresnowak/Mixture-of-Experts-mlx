import mlx.core as mx
import mlx.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    # Multi-head attention, implementation from https://arxiv.org/abs/1706.03762
    def __init__(self, emb_dim: int, num_heads: int, bias=True):
        # Dimension of model (emb_dim) is equal to num_heads * d_k, d_k = emb_dim / num_heads
        # emb_dim has to be divisble by num_heads

        # Q and K use the same d_k and we also say that d_v = d_k

        super().__init__()

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        d_k = emb_dim // num_heads
        self.scaling: float = 1 / np.sqrt(d_k)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=bias)

    def _split_heads(self, z: mx.array):
        batch, seq_len, _ = z.shape

        return z.reshape(
            batch, seq_len, self.num_heads, self.emb_dim // self.num_heads
        ).transpose(0, 2, 1, 3)  # (B, num_heads, seq_len, d_k)

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
        Q = self._split_heads(Q)
        K = self.W_k(x)
        K = self._split_heads(K)
        V = self.W_v(x)
        V = self._split_heads(V)

        def mask_fill(qk: mx.array) -> mx.array:
            if attn_mask is not None:
                mask = mx.where(attn_mask, 0, -float("inf"))
                qk = qk + mask.reshape(
                    1, 1, *attn_mask.shape
                )  # (B, num_heads, seq_len, seq_len), (1, 1, seq_len, seq_len)

            return qk

        score = (
            mask_fill(Q @ K.transpose(0, 1, 3, 2)) * self.scaling
        )  # (B, num_heads, seq_len, seq_len)
        attention = mx.softmax(score, axis=-1) @ V  # (B, num_heads, seq_len, d_k)

        multi_head = attention.transpose(
            0, 2, 1, 3
        ).reshape(
            batch, seq_len, self.emb_dim
        )  # (B, seq_len, H * d_k) # H * d_k = emb_dim. This is just the concatenation of each head’s d_k outputs. we haven’t yet “mixed” them into the model’s true embedding space, that happens in the final linear W_o.
        multi_head = self.W_o(multi_head)  # (B, seq_len, emb_dim)

        return multi_head


class GatedAttention(nn.Module):
    # Gated Attention, implementation from https://arxiv.org/abs/2505.06708
    def __init__(self, emb_dim: int, num_heads: int, bias=True):
        # Dimension of model (emb_dim) is equal to num_heads * d_k, d_k = emb_dim / num_heads
        # emb_dim has to be divisble by num_heads

        # Q and K use the same d_k and we also say that d_v = d_k

        super().__init__()

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        d_k = emb_dim // num_heads
        self.scaling: float = 1 / np.sqrt(d_k)

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.W_o = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.W_gates = nn.Linear(emb_dim, num_heads, bias=False) # head specific gating not elementwise

    def _split_heads(self, z: mx.array):
        batch, seq_len, _ = z.shape

        return z.reshape(
            batch, seq_len, self.num_heads, self.emb_dim // self.num_heads
        ).transpose(0, 2, 1, 3)  # (B, num_heads, seq_len, d_k)

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
        Q = self._split_heads(Q) # (B, num_heads, seq_len, d_k)
        K = self.W_k(x)
        K = self._split_heads(K)
        V = self.W_v(x)
        V = self._split_heads(V)

        def mask_fill(qk: mx.array) -> mx.array:
            if attn_mask is not None:
                mask = mx.where(attn_mask, 0, -float("inf"))
                qk = qk + mask.reshape(
                    1, 1, *attn_mask.shape
                )  # (B, num_heads, seq_len, seq_len), (1, 1, seq_len, seq_len)

            return qk

        score = (
            mask_fill(Q @ K.transpose(0, 1, 3, 2)) * self.scaling
        )  # (B, num_heads, seq_len, seq_len)
        attention = mx.softmax(score, axis=-1) @ V  # (B, num_heads, seq_len, d_k)

        attention = attention.transpose(
            0, 2, 1, 3
        ) # (batch, seq_len, h, d_k)

        # Apply the gating
        gate = mx.expand_dims(mx.sigmoid(self.W_gates(x)), axis=-1) # (batch, seq_len, h, 1), so each value we repeat it for all d_k
        attention_gating = attention * gate # (batch, seq_len, h, d_k)
        # this is query specific because we are applying it after the attention output, if it was applied to the Value and then from there we obtain the value
        # scores we would have the multiplication of the lower diagonal matrix times V, and we are having the results based on all the past values instead
        # (as we get the sum of that whole sequence length embeddings times the values)
        # gate = mx.expand_dims(mx.sigmoid(self.W_gates(x)), axis=-1) # (batch, seq_len, h, 1)
        # value_gating = V * gate # (batch, seq_len, h, d_v)
        # attention = attention @ value_gating
        attention_gating = attention_gating.reshape(batch, seq_len, self.emb_dim) # (batch, seq_len, emb_dim)


        multi_head = self.W_o(attention_gating)  # (B, seq_len, emb_dim)

        return multi_head
