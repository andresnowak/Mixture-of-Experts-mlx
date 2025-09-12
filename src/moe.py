from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .aux_losses import compute_expert_load_balance_loss
from .mlx_extension import one_hot


class TopKRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        shared_experts: int,
        routed_experts: int,
        top_k_routers: int,
    ):
        super().__init__()

        self.shared_experts = shared_experts
        self.routed_experts = routed_experts

        self.total_experts = shared_experts + routed_experts

        self.top_k_routers = top_k_routers

        self.expert_embeddings = nn.Linear(hidden_dim, self.total_experts, bias=False)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        TopK router

        Parameters
        ----------
        x : array
            Output form the Attention head (batch, seq_len, embed_dim).

        Returns
        -------
        array
            Affinity scores of the top_k_routers + shared experts to each token (batch, seq_len, top_k_routers + shared_experts)
        array
            Output indices saying which experts we will compute with (batch, seq_len, top_k_routers + shared_experts)
        """
        score_logits = self.expert_embeddings(x)  # (batch, seq_len, total_experts)

        score_logits[..., : self.shared_experts] = -float(
            "inf"
        )  # so shared experts don't affect the scores of the routed experts (they will go to 0)

        score_gate = mx.softmax(
            score_logits, axis=-1
        )  # u_t * e_i, (Batch, seq_len, total_experts)

        score_gate[..., : self.shared_experts] = 1.0

        top_experts_indices = mx.stop_gradient(
            mx.argpartition(
                -score_gate, kth=self.top_k_routers + self.shared_experts, axis=-1
            )
        )[..., : self.top_k_routers + self.shared_experts]

        return score_gate, top_experts_indices


class FFN(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int):
        super().__init__()

        self.ff_1 = nn.Linear(hidden_dim, ff_dim)
        self.ff_2 = nn.Linear(ff_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.ff_2(mx.maximum(self.ff_1(x), 0))


class MoE(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        shared_experts: int,
        routed_experts: int,
        top_k_routers: int,
    ):
        super().__init__()

        self.shared_experts = shared_experts
        self.routed_experts = routed_experts
        self.total_experts = shared_experts + routed_experts

        self.top_k_routers = top_k_routers

        self.router = TopKRouter(
            hidden_dim, shared_experts, routed_experts, top_k_routers
        )

        self.experts = [FFN(hidden_dim, ff_dim) for _ in range(self.total_experts)]

    def __call__(
        self, x: mx.array, return_load_balance_loss: bool = False
    ) -> mx.array | Tuple[mx.array, mx.array]:
        """
        MoE router

        Parameters
        ----------
        x : array
            Output form the Attention head (batch, seq_len, embed_dim).

        Returns
        -------
        array
            Output Tensor of form (batch, seq_len, embed_dim)
            Experts affinites scores of form (batch, seq_len, total_experts)
        """

        return self.__naive(x, return_load_balance_loss)

    def __naive(
        self, x: mx.array, return_load_balance_loss: bool = False
    ) -> mx.array | Tuple[mx.array, mx.array]:
        batch, seq_len, emb_dim = x.shape
        num_tokens = batch * seq_len

        experts_affinity, experts_indices = self.router(x)

        selected_experts_affinity = mx.take_along_axis(
            experts_affinity, experts_indices, axis=-1
        )

        routed_output = mx.zeros(
            (
                x.shape[0],
                x.shape[1],
                self.top_k_routers + self.shared_experts,
                x.shape[2],
            )
        )

        for e in range(self.total_experts):
            expert_indices = tuple(map(mx.array, np.where(experts_indices == e)))

            if expert_indices[0].size == 0:
                continue

            routed_input = mx.contiguous(
                x[*expert_indices[:-1]].reshape(-1, emb_dim)
            )  # gather

            out_e = self.experts[e](
                routed_input
            )  # (batch, seq_len_for_expert, embed_dim)

            routed_output[*expert_indices] = out_e  # scatter

        routed_output = (
            routed_output * mx.expand_dims(selected_experts_affinity, axis=-1)
        ).sum(axis=2)

        if return_load_balance_loss:
            load_balance_loss = compute_expert_load_balance_loss(
                self.shared_experts,
                self.routed_experts,
                self.top_k_routers,
                experts_affinity,
                experts_indices,
                num_tokens,
            )
            return routed_output, load_balance_loss

        return routed_output


class ExpertChoiceMoE(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_experts: int,
        capacity_factor: int,
        sequence_length: int,
        batch_size: int,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k_tokens = (
            batch_size * sequence_length * capacity_factor
        ) // num_experts  # n * c / e

        self.expert_embeddings = nn.Linear(hidden_dim, self.num_experts, bias=False)

        self.experts = [FFN(hidden_dim, ff_dim) for _ in range(self.num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Expert Choice MoE router

        Parameters
        ----------
        x : array
            Output form the Attention head (batch, seq_len, embed_dim).

        Returns
        -------
        array
            Output Tensor of form (batch, seq_len, embed_dim)
        """
        batch, seq_len, hidden_dim = x.shape

        x_input = x.reshape(-1, hidden_dim)

        logits = self.expert_embeddings(x_input)  # (N, E)
        affinity_scores = mx.softmax(
            logits, axis=-1
        )  # (N, E). Why do the softmax over the expert dimension and no the scores?

        chosen_tokens_indices = mx.stop_gradient(
            mx.argpartition(
                -affinity_scores.transpose(1, 0), kth=self.top_k_tokens, axis=-1
            )
        )[..., : self.top_k_tokens]  # (E, top_k_tokens)

        return self.__naive(
            x_input, chosen_tokens_indices, affinity_scores, batch * seq_len
        ).reshape(batch, seq_len, hidden_dim)

    def __naive(
        self,
        x: mx.array,
        chosen_tokens_indices: mx.array,
        affinity_scores: mx.array,
        total_tokens: int,
    ) -> mx.array:
        x_out = mx.zeros_like(x)

        for e in range(self.num_experts):
            output = self.experts[e](x[chosen_tokens_indices[e]]) # (top_k_tokens, emb_dim)

            x_out[chosen_tokens_indices[e]] += output * mx.expand_dims(affinity_scores[chosen_tokens_indices[e],e], axis=-1)

        return x_out


    def __parallel(
        self,
        x: mx.array,
        chosen_tokens_indices: mx.array,
        affinity_scores: mx.array,
        total_tokens: int,
    ) -> mx.array:
        P = one_hot(chosen_tokens_indices, total_tokens)  # (E, top_k_tokens, N)

        x_in = (
            P @ x
        )  # (E, top_k, N) @ (N, emb_dim) = (E, top_k, hidden_dim), one_hot matrix here is just a row chooser (so selecting the tokens it wants)

        expert_outputs = mx.stack(
            [self.experts[e](x_in[e]) for e in range(self.num_experts)]
        )  # (E, top_k, hidden_dim)

        # chosen_affinity_scores = affinity_score[chosen_tokens_indices.transpose(1, 0)]

        x_out = mx.einsum("ijl,li,ijd->ld", P, affinity_scores, expert_outputs)

        return x_out
