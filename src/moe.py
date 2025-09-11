from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# DeepSeekMoE: https://arxiv.org/abs/2401.06066


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim: int, shared_experts: int, routed_experts: int, top_k_routers: int):
        super().__init__()

        self.shared_experts = shared_experts
        self.routed_experts = routed_experts

        self.total_experts = shared_experts + routed_experts

        self.top_k_routers = top_k_routers

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
        score_logits = x @ self.expert_embeddings

        score_logits[..., :self.shared_experts] = -float("inf") # so shared experts don't affect the scores of the routed experts (they will go to 0)

        score_gate = mx.softmax(
            score_logits, axis=-1
        )  # u_t * e_i, (Batch, seq_len, total_experts)

        score_gate[..., :self.shared_experts] = 1.0

        top_experts_indices = mx.stop_gradient(mx.argpartition(-score_gate, kth=self.top_k_routers + self.shared_experts, axis=-1))[..., :self.top_k_routers + self.shared_experts]

        return score_gate, top_experts_indices

    def compute_load_balance_loss(self, gate_probs: mx.array, expert_indices: mx.array, num_tokens: int) -> mx.array:
        """
            Compute load balancing loss to encourage uniform expert usage.

            Parameters
            ----------
            gate_probs : array
                Raw gate probabilities (batch, seq_len, total_experts)
            expert_indices : array
                Selected expert indices (batch, seq_len, top_k_routers + shared_experts)
            num_tokens : int
                Total number of tokens in the batch

            Returns
            -------
            array
                Load balancing loss scalar
        """

        routed_gate_probs = gate_probs[..., self.shared_experts:]

        mean_gate_probs = routed_gate_probs.mean(axis=(0, 1)) # P_i in the paper

        experts_counts = mx.zeros(self.routed_experts)

        for e in range(self.shared_experts, self.total_experts):
            mask = (expert_indices == e).any(axis=-1) # (batch, seq_len)
            experts_counts[e] = mask.sum()

        experts_counts = experts_counts * (self.routed_experts / (self.top_k_routers * num_tokens)) # f_i in the paper

        # Load balancing loss: encourage P_i * f_i to be uniform
        load_balance_loss = (experts_counts * mean_gate_probs).sum()

        return load_balance_loss


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

    def __call__(self, x: mx.array, return_load_balance_loss: bool = False) -> mx.array | Tuple[mx.array, mx.array]:
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

        return self._naive_loop(x, return_load_balance_loss)


    def _naive_loop(self, x: mx.array, return_load_balance_loss: bool = False) -> mx.array | Tuple[mx.array, mx.array]:
        batch, seq_len, emb_dim = x.shape
        num_tokens = batch * seq_len

        experts_affinity, experts_indices = self.router(x)

        selected_experts_affinity = mx.take_along_axis(experts_affinity, experts_indices, axis=-1)

        routed_output = mx.zeros((x.shape[0], x.shape[1], self.top_k_routers + self.shared_experts, x.shape[2]))

        for e in range(self.top_k_routers + self.shared_experts):
            expert_indices = tuple(map(mx.array, np.where(experts_indices == e)))[:-1] # we don't care about the dimension of experts

            routed_input = mx.contiguous(x[*expert_indices].reshape(-1, emb_dim)) # gather

            out_e = self.experts[e](routed_input) # (batch, seq_len_for_expert, embed_dim)

            routed_output[:, :, e][*expert_indices] = out_e # scatter

        routed_output = (routed_output * mx.expand_dims(selected_experts_affinity, axis=-1)).sum(axis=2)

        if return_load_balance_loss:
            load_balance_loss = self.router.compute_load_balance_loss(
                           experts_affinity, experts_indices, num_tokens
            )
            return routed_output, load_balance_loss

        return routed_output

    def _parallel(self, x: mx.array, return_load_balance_loss: bool = False) -> mx.array | Tuple[mx.array, mx.array]:
        pass
