import torch
from torch import nn
from torch.nn import functional as F

from .config import UoEConfig


class _UoEExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_g = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_p = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, d_model, bias=False)


class UoELayer(nn.Module):
    def __init__(self, config: UoEConfig) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.n_routing_neurons = config.n_routing_neurons
        self.routed_experts = nn.ModuleList(
            _UoEExpert(config.d_model, config.expert_hidden_dim)
            for _ in range(config.n_routed_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)
        n_experts = len(self.routed_experts)
        n_route = self.n_routing_neurons

        routing_gate = torch.cat(
            [expert.w_g.weight[:n_route] for expert in self.routed_experts],
            dim=0,
        )
        routing_up = torch.cat(
            [expert.w_p.weight[:n_route] for expert in self.routed_experts],
            dim=0,
        )
        routing_hidden = F.silu(F.linear(flat_x, routing_gate)) * F.linear(
            flat_x, routing_up
        )
        routing_hidden = routing_hidden.reshape(
            flat_x.shape[0], n_experts, n_route
        )

        expert_scores = torch.linalg.vector_norm(routing_hidden.float(), dim=-1)
        selected_scores, expert_indices = expert_scores.topk(self.top_k, dim=-1)
        expert_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32)

        # The routing neurons contribute to both the shared and routed outputs.
        routing_scales = 1 + torch.zeros_like(expert_scores).scatter(
            -1, expert_indices, expert_weights
        )
        weighted_routing = routing_hidden * routing_scales.to(
            dtype=routing_hidden.dtype
        ).unsqueeze(-1)
        routing_out = torch.cat(
            [expert.w_o.weight[:, :n_route] for expert in self.routed_experts],
            dim=1,
        )
        output = F.linear(weighted_routing.flatten(start_dim=1), routing_out)

        # Run only the remaining neurons for tokens assigned to each expert.
        for expert_id, expert in enumerate(self.routed_experts):
            token_indices, topk_positions = torch.where(expert_indices == expert_id)
            if token_indices.numel() == 0:
                continue

            expert_input = flat_x.index_select(0, token_indices)
            gate = F.silu(F.linear(expert_input, expert.w_g.weight[n_route:]))
            up = F.linear(expert_input, expert.w_p.weight[n_route:])
            expert_output = F.linear(gate * up, expert.w_o.weight[:, n_route:])
            weights = expert_weights[token_indices, topk_positions]
            weighted_output = expert_output * weights.to(
                dtype=expert_output.dtype
            ).unsqueeze(-1)
            output.index_add_(0, token_indices, weighted_output)

        return output.reshape_as(x), expert_scores.reshape(
            batch_size, seq_len, n_experts
        )
