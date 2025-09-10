import mlx.core as mx
import mlx.nn as nn

import numpy as np

from typing import Union, Tuple

from .mlx_extension import multinomial
from .moe import MoE, FFN

# https://arxiv.org/abs/1706.03762, but we use pre-norm and dropout


class MultiHeadAttention(nn.Module):
    # Multi-head attention
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ff_dim: int,
        num_heads: int,
        ff_function: Union[FFN, MoE],
        prob: float = 0.5,
    ):
        # ff_dim commonly is 4 times the size of emb_dim
        super().__init__()

        self.attn_block = MultiHeadAttention(emb_dim, num_heads)

        self.ff = ff_function
        self.dropout = nn.Dropout(prob)

        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)

    def __call__(self, x: mx.array, attn_mask: mx.array | None = None, return_aux_loss: bool = False) -> mx.array | Tuple[mx.array, mx.array]:
        """
        Transformer block

        Parameters
        ----------
        x : array
            Input sequence of shape (batch, seq_len, embed_dim).

        Returns
        -------
        array
            Output tensor of shape  (batch, seq_len, embed_dim)
            Load balance loss of MoE if return_aux_loss is true (batch, seq_len, total_experts)
        """

        attention = self.attn_block(self.norm_1(x), attn_mask)
        x = x + attention

        aux_loss = mx.array(0.0)

        if isinstance(self.ff, MoE) and return_aux_loss:
            ff, aux_loss = self.ff(self.norm_2(x), return_aux_loss)
        else:
            ff = self.ff(self.norm_2(x))
            if isinstance(self.ff, FFN):
                ff = self.dropout(ff)

        x = x + ff  # (Batch, seq_len, emb_dim)

        if return_aux_loss:
            return x, aux_loss

        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_dim: int,
        emb_dim: int,
        num_heads: int,
        layers: int,
        ff_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.pos_embedding = nn.init.he_normal()(mx.zeros((max_len, emb_dim)))

        self.transformer_blocks = [
            TransformerBlock(emb_dim, ff_dim, num_heads, FFN(emb_dim, ff_dim), 0.5)
            for i in range(layers)
        ]

        self.proj_ff = nn.Linear(emb_dim, vocab_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Decoder Transformer

        Parameters
        ----------
        x : array
            Input sequence of shape (batch, seq_len, 1).

        Returns
        -------
        array
            The raw logits output tensor of shape (batch, seq_len, vocab_dim).
        """

        assert len(x.shape) == 3

        _, seq_len, _ = x.shape

        attn_mask = mx.tril(mx.ones((seq_len, seq_len)), k=0)

        embedding = self.embedding(x.squeeze(axis=-1))
        pos_embedding = self.pos_embedding[:seq_len, :]

        x = embedding + pos_embedding

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask)

        x = self.proj_ff(x)

        return x  # raw logits in the form (batch, seq_len, vocab_dim)

    def generate(
        self,
        sequence: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ):
        """
        Decoder Transformer

        Parameters
        ----------
        sequence : array
            Input sequence of shape (batch=1, seq_len).

        Returns
        -------
        array
            The raw logits output tensor of shape (batch=1, seq_len).
        """

        return generate(self, sequence, max_new_tokens, temperature, do_sample, top_k)


class MoEDecoderTransformer(nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_dim: int,
        emb_dim: int,
        num_heads: int,
        ff_dim: int,
        shared_experts: int,
        routed_experts: int,
        top_k_routers: int,
        layers: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.pos_embedding = nn.init.he_normal()(mx.zeros((max_len, emb_dim)))

        self.transformer_blocks = [
            TransformerBlock(
                emb_dim, ff_dim, num_heads, MoE(emb_dim, ff_dim, shared_experts, routed_experts, top_k_routers), 0.5
            )
            for i in range(layers)
        ]

        self.proj_ff = nn.Linear(emb_dim, vocab_dim, bias=False)

    def __call__(self, x: mx.array, return_aux_loss: bool = False) -> mx.array | Tuple[mx.array, mx.array]:
        """
        MoE Decoder Transformer

        Parameters
        ----------
        x : array
            Input sequence of shape (batch, seq_len, 1).

        Returns
        -------
        array
            The raw logits output tensor of shape (batch, seq_len, vocab_dim).
        """

        assert len(x.shape) == 3

        _, seq_len, _ = x.shape

        attn_mask = mx.tril(mx.ones((seq_len, seq_len)), k=0)

        embedding = self.embedding(x.squeeze(axis=-1))
        pos_embedding = self.pos_embedding[:seq_len, :]

        x = embedding + pos_embedding

        total_aux_loss = mx.array(0.0)

        for transformer_block in self.transformer_blocks:
            if return_aux_loss:
                x, aux_loss = transformer_block(x, attn_mask,return_aux_loss)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                x = transformer_block(x, attn_mask)

        x = self.proj_ff(x) # raw logits in the form (batch, seq_len, vocab_dim)

        if return_aux_loss:
            return x, total_aux_loss

        return x

    def generate(
        self,
        sequence: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
    ):
        """
        Decoder Transformer

        Parameters
        ----------
        sequence : array
            Input sequence of shape (batch=1, seq_len).

        Returns
        -------
        array
            The raw logits output tensor of shape (batch=1, seq_len).
        """

        return generate(self, sequence, max_new_tokens, temperature, do_sample, top_k)


def generate(
    transformer: nn.Module,
    sequence: mx.array,
    max_new_tokens: int,
    temperature: float = 1.0,
    do_sample: bool = False,
    top_k: int | None = None,
):
    """
    Decoder Transformer

    Parameters
    ----------
    sequence : array
        Input sequence of shape (batch=1, seq_len).

    Returns
    -------
    array
        The raw logits output tensor of shape (batch=1, seq_len).
    """

    starting_len = sequence.shape[-1]

    sequence = mx.expand_dims(sequence, -1)

    for _ in range(max_new_tokens - starting_len):
        logits = transformer(sequence)  # (batch=1, seq_len, vocab_dim)

        if top_k is not None:
            top_logits = mx.topk(
                logits, k=top_k, axis=-1
            )  # only returns the values not the positions

            logits = mx.where(logits < top_logits[..., -1:], -float("inf"), logits)

        probs = mx.softmax(
            logits[:, -1, :] / temperature, axis=-1
        )  # we only want the last logit for the generation (this is our target)

        if do_sample:
            idx_next = multinomial(probs, num_samples=1)
        else:
            idx_next = probs.argmax(axis=-1, keepdims=True)

        sequence = mx.expand_dims(
            mx.concat([sequence.squeeze(-1), idx_next], axis=-1), -1
        )

    return sequence.squeeze(-1)
