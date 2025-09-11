import mlx.core as mx


def compute_expert_load_balance_loss(
    shared_experts: int,
    routed_experts: int,
    top_k_routers: int,
    gate_probs: mx.array,
    expert_indices: mx.array,
    num_tokens: int,
) -> mx.array:
    """
    Compute Expert load balancing loss to encourage uniform expert usage.

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
    # Implemented as in DeepSeekMoE: https://arxiv.org/abs/2401.06066

    routed_gate_probs = gate_probs[..., shared_experts:]

    mean_gate_probs = routed_gate_probs.mean(axis=(0, 1))  # P_i in the paper

    experts_counts = mx.zeros(routed_experts)

    for e in range(shared_experts, shared_experts + routed_experts):
        e_idx = e - shared_experts
        mask = (expert_indices == e).any(axis=-1)  # (batch, seq_len)
        experts_counts[e_idx] = mask.sum()

    experts_counts = experts_counts * (
        routed_experts / (top_k_routers * num_tokens)
    )  # f_i in the paper

    # Load balancing loss: encourage P_i * f_i to be uniform
    load_balance_loss = (experts_counts * mean_gate_probs).sum()

    return load_balance_loss
