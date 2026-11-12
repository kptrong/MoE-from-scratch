import torch
from torch import nn
from torch.nn import functional as F

class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, expert_hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, expert_hidden_dim, bias = False)
        self.up_proj = nn.Linear(d_model, expert_hidden_dim, bias = False)
        self.down_proj = nn.Linear(expert_hidden_dim, d_model, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.square().mean(dim=-1, keepdim=True)
        normalized = x / torch.sqrt(mean_square + self.eps)
        return self.weight * normalized


    