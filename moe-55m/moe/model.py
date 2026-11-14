import torch
from torch import nn

from .aoe import AoELayer
from .attention import CausalSelfAttention
from .config import AoEConfig, ModelConfig
from .layers import RMSNorm
from .moe_layer import MoELayer

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig | AoEConfig,) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim = config.d_model, eps = 1e-5)
        self.attention = CausalSelfAttention(config)

        self.moe_norm = RMSNorm(dim = config.d_model, eps = 1e-5)
        self.moe = (
            AoELayer(config) if isinstance(config, AoEConfig) else MoELayer(config)
        )

    def forward(self, x: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attn_norm(x)) # Residual connection for attention 

        moe_output, router_logits = self.moe(self.moe_norm(x)) # MoE layer
        return x + moe_output, router_logits

class MoELanguageModel(nn.Module):
    def __init__(self, config: ModelConfig | AoEConfig,) -> None:
        super().__init__()

        # Khởi tạo embedding cho các token trong từ điển
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(dim = config.d_model, eps = 1e-5)

        # Đầu ra dự đoán xác suất cho từng token trong từ điển    
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False,)

        # Khởi tạo trọng số cho mô hình
        self.apply(self._init_weights)

        # Dùng cùng một ma trận trọng số cho lớp embedding và lớp đầu ra dự đoán xác suất
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Khởi tạo trọng số cho các lớp nn.Embedding và nn.Linear bằng phân phối chuẩn với trung bình = 0 và độ lệch chuẩn = 0.02
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.token_embedding(input_ids)
        router_logits_per_layer = []

        for block in self.blocks:
            x, router_logits = block(x)
            router_logits_per_layer.append(router_logits)

        x = self.final_norm(x)
        # logits: tensor dự đoán xác suất cho từng token trong từ điển, có kích thước (batch_size, seq_len, vocab_size)
        # Router logits (MoE) or L2 scores (AoE), in block order.
        logits = self.lm_head(x)

        return logits, tuple(router_logits_per_layer)