import mlx.core as mx

from .mlx_extension import one_hot


def compute_expert_load_balance_loss(
    shared_experts: int,
    routed_experts: int,
    top_k_routers: int,
    gate_probs: mx.array,
    expert_indices: mx.array,
    num_tokens: int,
) -> mx.array:
    """
    Compute Expert load balancing loss to encourage uniform expert usage. As in DeepSeekMoE: https://arxiv.org/abs/2401.06066

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

    routed_gate_probs = gate_probs[..., shared_experts:]

    mean_gate_probs = routed_gate_probs.mean(axis=(0, 1))  # P_i in the paper

    experts_mask = one_hot(
        expert_indices, shared_experts + routed_experts
    )  # (batch, seq_len, selected_experts, total_experts)

    experts_counts = experts_mask.sum(axis=(0, 1, 2))[
        shared_experts:
    ]  # shared experts are always the first ones. (num_experts)

    experts_counts = experts_counts * (
        routed_experts / (top_k_routers * num_tokens)
    )  # f_i in the paper

    # Load balancing loss: encourage P_i * f_i to be uniform
    load_balance_loss = (experts_counts * mean_gate_probs).sum()

    return load_balance_loss


def compute_load_balance_loss(
    num_experts: int,
    gate_probs: mx.array,
    expert_indices: mx.array,
    num_tokens: int,
) -> mx.array:
    """
    Compute load balancing loss to encourage uniform expert usage. As in switch transformer https://arxiv.org/abs/2101.03961

    Parameters
    ----------
    gate_probs : array
        Raw gate probabilities (batch, seq_len, total_experts)
    expert_indices : array
        Selected expert indices (batch, seq_len, top_k_routers)
    num_tokens : int
        Total number of tokens in the batch

    Returns
    -------
    array
        Load balancing loss scalar
    """

    mean_gate_probs = gate_probs.mean(axis=(0, 1))  # P_i in the paper

    experts_mask = one_hot(
        expert_indices, num_experts
    )  # (batch, seq_len, selected_experts, num_experts)

    experts_counts = experts_mask.sum(axis=(0, 1, 2)) # (num_experts)

    experts_counts = experts_counts / num_tokens  # f_i in the paper

    # Load balancing loss: encourage P_i * f_i to be uniform
    load_balance_loss = (experts_counts * mean_gate_probs).sum()

    return load_balance_loss
