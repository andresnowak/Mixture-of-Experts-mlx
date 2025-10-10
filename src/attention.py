import mlx.core as mx
import mlx.nn as nn
import numpy as np
from torch import topk

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

class DeepSeekSparseAttention(MultiHeadAttention):
    # DSA from DeepSeek v3.2, implementation from https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    def __init__(self, max_seq_len: int, emb_dim: int, num_heads: int, bias: bool=True, use_rope: bool=False):
        super().__init__(max_seq_len, emb_dim, num_heads, bias, use_rope)

        # Note: in DSA the index weights can have a different emb_dim (so different head_dim using index_emb_dim)

        self.index_n_heads: int = num_heads # NOTE: This doesn't ahve to be the same as the amount of heads in attention
        self.index_topk: int = max_seq_len // 2 # NOTE: This part I don't know what topk should be used
        self.index_head_dim = self.emb_dim // self.index_n_heads
        self.wq = nn.Linear(self.emb_dim, self.index_n_heads * self.index_head_dim, bias=True)
        self.wk = nn.Linear(self.emb_dim, self.index_head_dim, bias=True) # one for all heads
        self.weights_proj = nn.Linear(self.emb_dim, self.index_n_heads, bias=True)

    def _indexer(self, x: mx.array) -> mx.array:
        # x has shape (batch_size, seq_len, emb_dim)

        # NOTE: Missing here also partially applying Rope

        # 1. Calculate q, k, and w
        # Note the fix: use self.wk for keys
        q = self.wq(x)  # Shape: (batch, seq_len, n_heads * index_head_dim)
        k = self.wk(x)  # Shape: (batch, seq_len, index_head_dim)
        w = self.weights_proj(x) # Shape: (batch, seq_len, index_n_heads)

        # 2. Reshape for multi-head processing
        # Reshape q to separate the heads
        batch, seq_len, _ = q.shape
        q = q.reshape(batch, seq_len, self.index_n_heads, self.index_head_dim) # (batch, seq_len, index_n_heads, index_head_dim)
        q = q.transpose(0, 2, 1, 3) # (batch, index_n_heads, seq_len, index_head_dim)
        score = nn.relu(q @ mx.expand_dims(k.transpose(-2, -1), axis=1)) # (batch, index_n_heads, seq_len, seq_len)

        indexer_score = (mx.expand_dims(w.transpose(-2, -1), axis=-1) * score).sum(axis=1) # (batch, seq_len, seq_len)

        return indexer_score


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

        Q = self._split_heads(Q) # (B, num_heads, seq_len, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        indexer_scores = self._indexer(x)
        indexer_scores = self._apply_mask(indexer_scores, attn_mask) # because we don't want to choose future tokens
        topk_indices = mx.stop_gradient(mx.argpartition(indexer_scores, -self.index_topk, axis=-1)[:, :, -self.index_topk:]) # (batch, seq_len, top_k)


        # NOTE: this is just sparse attention, we are not reducintg complexity here from O(L^2) to O(Lk)
        # topk_indices = mx.expand_dims(topk_indices, axis=1) # (batch, 1, seq_len, top_k)

        # index_mask: mx.array = mx.full(shape=(batch, seq_len seq_len), vals=-float("inf"), dtype=x.dtype)
        # index_mask[:, topk_indices] = 0.0
        # index_mask += attn_mask # now the masking is not triangluar, it now also in each row (so the 0 until present, and then everything else masked) now also has values that are masked in between

        # attention = self._compute_attention(Q, K, V, index_mask)

        topk_indices = mx.expand_dims(topk_indices, axis=(1, -1)) # (batch, 1, seq_len, top_k, 1)
        topk_indices = mx.repeat(topk_indices, Q.shape[-1], axis=-1)
        topk_indices = mx.repeat(topk_indices, Q.shape[1], axis=1) # (batch, num_heads, seq_len, top_k, D_k)

        K_expanded = mx.expand_dims(K, axis=3) # (batch, num_heads, seq_len, 1, D_k)
        V_expanded = mx.expand_dims(V, axis=3) # (batch, num_heads, seq_len, 1, D_k)

        # For each of the L queries, we now select k keys.
        K_sparse = mx.take_along_axis(K_expanded, topk_indices, axis=2)
        V_sparse = mx.take_along_axis(V_expanded, topk_indices, axis=2)
        # K_sparse/V_sparse shape: (batch, num_heads, seq_len, 1, D_k)

        # 5. Compute attention only on the sparse sets of K and V.
        # Reshape Q for batched matmul with the sparse keys.
        Q_reshaped = mx.expand_dims(Q, axis=3) # Shape: (B, H, L, 1, D_k)

        attention = self._compute_attention(Q_reshaped, K_sparse, V_sparse, None)

        return self._post_attention_processing(attention, x)


    def _compute_attention(self, Q: mx.array, K: mx.array, V: mx.array, attn_mask: mx.array | None) -> mx.array:
        """Compute attention scores and apply to values. Can be overridden by subclasses."""

        # Q @ K_sparse^T
        # Cost O(Lk)
        sparse_scores = (Q @ K.transpose(0, 1, 2, 4, 3)).squeeze(3) # Shape: (B, H, L, k)
        sparse_scores = self._apply_mask(sparse_scores, attn_mask) * self.scaling

        # Softmax is now over k elements, not L elements.
        weights = mx.softmax(sparse_scores, axis=-1)

        # weights @ V_sparse
        # cost O(Lk)
        attention = (mx.expand_dims(weights, axis=3) @ V).squeeze(3) # Shape: (B, H, L, D_k)
        return attention
