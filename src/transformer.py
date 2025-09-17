import mlx.core as mx
import mlx.nn as nn

import numpy as np

from typing import Union, Tuple, Dict, Any

from .mlx_extension import multinomial
from .moe import ExpertChoiceMoE, MoE, FFN
from .positional_embeddings import sinusoidal_embeddings, absolute_embeddings
from .attention import MultiHeadAttention, GatedAttention

# https://arxiv.org/abs/1706.03762, but we use pre-norm and dropout


class TransformerBlock(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        emb_dim: int,
        num_heads: int,
        ff_function: Union[FFN, MoE, ExpertChoiceMoE],
        attention_type: str = "MultiHeadAttention",
        prob: float = 0.5,
        use_rope: bool = False
    ):
        # ff_dim commonly is 4 times the size of emb_dim
        super().__init__()

        self.attn_block: None | MultiHeadAttention | GatedAttention = None

        if attention_type == "MultiHeadAttention":
            self.attn_block = MultiHeadAttention(max_seq_len, emb_dim, num_heads, use_rope=use_rope)
        elif attention_type == "GatedAttention":
            self.attn_block = GatedAttention(max_seq_len, emb_dim, num_heads, use_rope=use_rope)
        else:
            raise ValueError(f"Incorrect attention type: {attention_type}")

        self.ff = ff_function
        self.dropout = nn.Dropout(prob)

        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)

    def __call__(
        self,
        x: mx.array,
        attn_mask: mx.array | None = None,
        return_aux_loss: bool = False,
    ) -> mx.array | Tuple[mx.array, mx.array]:
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

        ff = self.dropout(ff)

        x = x + ff  # (Batch, seq_len, emb_dim)

        if return_aux_loss:
            return x, aux_loss

        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        **config,
    ):
        super().__init__()

        ff_function_type: str = config["ff_function"]
        batch_size: int = config["batch_size"]
        vocab_dim: int = config["vocab_dim"]
        emb_dim: int = config["emb_dim"]
        routing_type: str | None = config.get("routing_type", None)
        max_len: int = config["max_len"]
        ff_dim: int = config["ff_dim"]
        num_experts: int = config.get("num_experts", 0)
        shared_experts: int = config.get("shared_experts", 0)
        top_k_routers: int = config.get("top_k_routers", 0)
        capacity_factor: int = config.get("capacity_factor", 0)
        layers: int = config["layers"]
        num_heads: int = config["num_heads"]
        pos_embedding_type: str = config.get("pos_embedding_type", "absolute")
        attention_type: str = config.get("attention_type", "MultiHeadAttention")

        routed_experts = num_experts - shared_experts

        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.pos_embedding: None | mx.array = None
        self.use_rope = False

        if pos_embedding_type == "absolute":
            self.pos_embedding = absolute_embeddings(max_len, emb_dim)
        elif pos_embedding_type == "sinusoidal":
            self.pos_embedding = sinusoidal_embeddings(max_len, emb_dim)
        elif pos_embedding_type == "RoPE":
            self.use_rope = True
        else:
            raise ValueError(f"Incorrect type of positional embedding {pos_embedding_type}")

        def make_ff_function() -> Union[FFN, MoE, ExpertChoiceMoE]:
            ff_function: Union[None, FFN, MoE, ExpertChoiceMoE] = None
            if ff_function_type == "MoEDecoderTransformer":
                if routing_type == "MoE":
                    ff_function = MoE(
                        emb_dim, ff_dim, shared_experts, routed_experts, top_k_routers
                    )
                elif routing_type == "ExpertChoiceMoE":
                    ff_function = ExpertChoiceMoE(
                        emb_dim,
                        ff_dim,
                        routed_experts,
                        capacity_factor,
                        batch_size,
                        max_len,
                    )
                else:
                    raise ValueError(f"Incorrect routing type {routing_type}")
            elif ff_function_type == "DecoderTransformer":
                ff_function = FFN(emb_dim, ff_dim)
            else:
                raise ValueError(f"Incorrect FFN function type {ff_function}")

            return ff_function

        self.transformer_blocks = [
            TransformerBlock(max_seq_len=max_len, emb_dim=emb_dim, num_heads=num_heads, ff_function=make_ff_function(), attention_type=attention_type, prob=0.5, use_rope=self.use_rope)
            for _ in range(layers)
        ]

        self.proj_ff = nn.Linear(emb_dim, vocab_dim, bias=False)

    def __call__(
        self, x: mx.array, return_aux_loss: bool = False
    ) -> mx.array | Tuple[mx.array, mx.array]:
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
        x = embedding

        if not self.use_rope:
            pos_embedding = self.pos_embedding[:seq_len, :]
            x += pos_embedding

        total_aux_loss = mx.array(0.0)

        for transformer_block in self.transformer_blocks:
            if return_aux_loss:
                x, aux_loss = transformer_block(x, attn_mask, return_aux_loss)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                x = transformer_block(x, attn_mask)

        x = self.proj_ff(x)  # raw logits in the form (batch, seq_len, vocab_dim)

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
