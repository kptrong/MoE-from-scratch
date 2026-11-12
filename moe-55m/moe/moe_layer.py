import torch
from torch import nn
from .config import ModelConfig
from .layers import SwiGLUExpert
from .router import TopKRouter

class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.router = TopKRouter(
            d_model = config.d_model,
            n_routed_experts = config.n_routed_experts,
            top_k = config.top_k,
        )

        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(
                d_model = config.d_model,
                expert_hidden_dim = config.expert_hidden_dim,
            )
            for _ in range(config.n_routed_experts)
        ])

        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(
                d_model = config.d_model,
                expert_hidden_dim = config.expert_hidden_dim,
            )
            for _ in range(config.n_shared_experts)
        ])

    def forward(self, x: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape # [2, 3, 512]
        flat_x = x.view(-1, original_shape[-1]) # [6, 512]
        expert_indices, expert_weights, router_logits = self.router(flat_x)
        # expert_indices: [6, 3], expert_weights: [6, 3], router_logits: [6, 9]
        output = torch.zeros_like(flat_x) # [6, 512]

        for expert_id, expert in enumerate(self.routed_experts):
            # Tìm các chỉ số của các token được định tuyến đến chuyên gia hiện tại
            token_indices, topk_positions = torch.where(
                expert_indices == expert_id
            )
            if token_indices.numel() == 0:
                continue

            expert_input = flat_x.index_select(0, token_indices) 
            expert_output = expert(expert_input)

            routing_weights = expert_weights[
                token_indices, topk_positions
            ].to(dtype = expert_output.dtype)

            weighted_output = expert_output * routing_weights.unsqueeze(-1)
            output.index_add_(
                0,
                token_indices, 
                weighted_output.to(dtype = output.dtype),
            )
        
        for expert in self.shared_experts:
            output = output + expert(flat_x)

        logits_shape = original_shape[:-1] + (router_logits.shape[-1],)

        return (
            output.reshape(original_shape),
            router_logits.reshape(logits_shape),
        )


