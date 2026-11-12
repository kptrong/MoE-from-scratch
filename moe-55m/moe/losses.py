import torch
from torch.nn import functional as F

def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor,) -> torch.Tensor:

    # input_ids: [B, T], logits: [B, T, V]
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError(
            "Expected logits [B, T, V] and input_ids [B, T]"
        )

    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must have matching B and T")

    if input_ids.shape[1] < 2:
        raise ValueError("Sequence length must be at least 2")

    if logits.shape[0] == 0 or logits.shape[-1] == 0:
        raise ValueError("Batch size and vocabulary size must be positive")


    prediction_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    targets = input_ids[:, 1:].reshape(-1)
    
    return F.cross_entropy(prediction_logits, targets)

def load_balancing_loss(router_logits_per_layer: tuple[torch.Tensor, ...], top_k: int,) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(router_logits_per_layer, tuple)
        or len(router_logits_per_layer) == 0
    ):
        raise ValueError("router_logits_per_layer must be a non-empty tuple")

    if any(
        not isinstance(logits, torch.Tensor) or logits.ndim != 3
        for logits in router_logits_per_layer
    ):
        raise ValueError("Each router logits tensor must have shape [B, S, E]")

    first_logits = router_logits_per_layer[0]
    n_experts = first_logits.shape[-1]

    if type(top_k) is not int or not 1 <= top_k <= n_experts:
        raise ValueError(
            "top_k must be an integer in [1, n_routed_experts]"
        )

    layer_losses = []
    counts_per_layer = []

    for layer_index, logits in enumerate(router_logits_per_layer):
        if logits.shape[-1] != n_experts:
            raise ValueError("All layers must have the same number of experts")

        if logits.device != first_logits.device:
            raise ValueError("All router logits must be on the same device")

        flat_logits = logits.reshape(-1, n_experts)
        n_tokens = flat_logits.shape[0]

        if n_tokens == 0:
            raise ValueError(f"Layer {layer_index} must contain at least one token")

        probabilities = F.softmax(
            flat_logits,
            dim=-1,
            dtype=torch.float32,
        )
        mean_probabilities = probabilities.mean(dim=0)

        expert_indices = torch.topk(
            flat_logits,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices

        counts = torch.bincount(
            expert_indices.reshape(-1),
            minlength=n_experts,
        ).detach()

        selection_fractions = counts.to(
            dtype=mean_probabilities.dtype
        ) / (n_tokens * top_k)

        layer_loss = n_experts * (
            selection_fractions * mean_probabilities
        ).sum()

        layer_losses.append(layer_loss)
        counts_per_layer.append(counts)

    aux_loss = torch.stack(layer_losses).mean()
    expert_counts = torch.stack(counts_per_layer).detach()

    return aux_loss, expert_counts
