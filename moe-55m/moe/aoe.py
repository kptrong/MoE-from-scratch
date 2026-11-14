import torch
from torch import nn
from torch.nn import functional as F

from .config import AoEConfig


class AoEExpert(nn.Module):
    def __init__(self, d_model: int, aoe_rank: int, expert_hidden_dim: int) -> None:
        super().__init__()
        self.w_down = nn.Linear(d_model, aoe_rank, bias=False)
        self.w_up = nn.Linear(aoe_rank, expert_hidden_dim, bias = False)
        self.w_p = nn.Linear(d_model, expert_hidden_dim, bias = False)
        self.w_o = nn.Linear(expert_hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.w_o(F.silu(self.w_up(c)) * self.w_p(x))


class AoELayer(nn.Module):
    def __init__(self, config: AoEConfig) -> None:
        super().__init__()
        if config.n_shared_experts != 0:
            raise ValueError("AoELayer requires n_shared_experts=0")

        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.top_k
        self.aoe_rank = config.aoe_rank

        self.experts = nn.ModuleList([
            AoEExpert(config.d_model, config.aoe_rank, config.expert_hidden_dim)
            for _ in range(self.n_routed_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape

        # Gộp trọng số w_down của mọi expert để chiếu trong một phép tính.
        w_down = torch.cat([expert.w_down.weight for expert in self.experts], dim=0)
        activations = F.linear(x, w_down).reshape(
            batch_size, seq_len, self.n_routed_experts, self.aoe_rank
        )

        expert_scores = torch.linalg.vector_norm(activations, dim=-1)
        selected_scores, expert_indices = expert_scores.topk(self.top_k, dim=-1)
        expert_weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32)

        flat_x = x.reshape(-1, d_model)
        flat_activations = activations.reshape(
            -1, self.n_routed_experts, self.aoe_rank
        )
        expert_indices = expert_indices.reshape(-1, self.top_k)
        expert_weights = expert_weights.reshape(-1, self.top_k)
        output = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_indices, topk_positions = torch.where(expert_indices == expert_id)
            if token_indices.numel() == 0:
                continue

            expert_input = flat_x.index_select(0, token_indices)
            c = flat_activations[token_indices, expert_id]
            expert_output = expert(expert_input, c)

            weights = expert_weights[token_indices, topk_positions]
            weighted_output = expert_output * weights.to(expert_output.dtype).unsqueeze(-1)
            output.index_add_(0, token_indices, weighted_output.to(output.dtype))

        return output.reshape_as(x), expert_scores
