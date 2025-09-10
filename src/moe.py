from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim: int, shared_experts: int, routed_experts: int, top_k_routers: int):
        super().__init__()

        self.shared_experts = shared_experts
        self.routed_experts = routed_experts

        self.total_experts = shared_experts + routed_experts

        self.top_k_routers = top_k_routers + shared_experts

        self.expert_embeddings = mx.random.uniform(shape=(hidden_dim, self.total_experts))

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
        score_gate = mx.softmax(
            x @ self.expert_embeddings, axis=-1
        )  # u_t * e_i, (Batch, seq_len, total_experts)

        score_gate[..., :self.shared_experts] = 1.0

        top_experts_indices = mx.stop_gradient(mx.argpartition(-score_gate, kth=self.top_k_routers, axis=-1))[..., :self.top_k_routers]

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
        self, hidden_dim: int, ff_dim: int, shared_experts: int, routed_experts: int, top_k_routers: int
    ):
        super().__init__()

        self.shared_experts = shared_experts
        self.routed_experts = routed_experts
        self.total_experts = shared_experts + routed_experts

        self.top_k_routers = top_k_routers

        self.router = TopKRouter(hidden_dim, shared_experts, routed_experts, top_k_routers)

        self.experts = [FFN(hidden_dim, ff_dim) for _ in range(self.total_experts)]

    def __call__(self, x: mx.array) -> mx.array:
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
        """

        batch, seq_len, emb_dim = x.shape

        expert_affinity, experts_indices = self.router(x)

        expert_affinity = mx.take_along_axis(expert_affinity, experts_indices, axis=-1)

        routed_output = mx.zeros((x.shape[0], x.shape[1], self.top_k_routers + self.shared_experts, x.shape[2]))

        for e in range(self.top_k_routers + self.shared_experts):
            expert_indices = tuple(map(mx.array, np.where(experts_indices == e)))[:-1] # we don't care about the dimension of experts

            routed_input = mx.contiguous(x[*expert_indices].reshape(-1, emb_dim)) # gather

            out_e = self.experts[e](routed_input) # (batch, seq_len_for_expert, embed_dim)

            routed_output[:, :, e][*expert_indices] = out_e # scatter

        routed_output = (routed_output * mx.expand_dims(expert_affinity, axis=-1)).sum(axis=2)

        return routed_output
