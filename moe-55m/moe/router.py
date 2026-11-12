import torch
from torch import nn
from torch.nn import functional as F

class TopKRouter(nn.Module):
    def __init__(self, d_model: int, n_routed_experts: int, top_k: int) -> None:
        super().__init__() 
        
        arguments = {
            "d_model": d_model,
            "n_routed_experts": n_routed_experts,
            "top_k": top_k,
        }
        # Kiểm tra các trường phải là số nguyên dương
        for name, value in arguments.items():
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

        if top_k > n_routed_experts:
            raise ValueError("top_k must not exceed n_routed_experts")

        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_routed_experts, bias = False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(x)

        selected_logits, expert_indices = torch.topk(
            router_logits,
            k = self.top_k,
            dim = -1,
            largest = True,
            sorted = True,
        )

        expert_weights = F.softmax(
            selected_logits, 
            dim = -1, 
            dtype = torch.float32,
        )

        # Trả về các chỉ số của chuyên gia được chọn, trọng số của chúng và các logits ban đầu từ bộ định tuyến
        return expert_indices, expert_weights, router_logits


    




