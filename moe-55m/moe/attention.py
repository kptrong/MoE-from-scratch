import torch
import math
from torch import nn
from .config import AoEConfig, ModelConfig


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0,) -> None:
        super().__init__()
        # Kiểm tra tính hợp lệ của head_dim
        if type(head_dim) is not int or head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")

        # Kiểm tra tính hợp lệ của base
        # base dùng để xác định tần số xoay cho các vị trí trong embedding. Nó phải là một số dương hữu hạn.
        if (
            isinstance(base, bool)
            or not isinstance(base, (int, float))
            or not math.isfinite(base)
            or base <= 0
        ):
            raise ValueError("base must be a positive finite number")

        # Các số mũ tương ứng -2*j/head_dim.
        pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        
        # Tính toán tần số nghịch đảo cho các vị trí trong embedding. Công thức là base ** (-2 * j / head_dim), với j là chỉ số của cặp.
        inv_freq = float(base) ** (-pair_indices / head_dim)
        
        # Lưu trữ tần số nghịch đảo dưới dạng buffer để không được coi là tham số học. Buffer này sẽ được sử dụng trong quá trình tính toán embedding xoay.
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,) -> torch.Tensor:
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        a = x[..., 0::2]
        b = x[..., 1::2]

        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        # Ghép lại theo thứ tự a0, b0, a1, b1, ...
        return torch.stack(
            (rotated_a, rotated_b),
            dim=-1,
        ).flatten(start_dim=-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]

        positions = torch.arange(
            seq_len,
            device=q.device,
            dtype=torch.float32,
        )
        
        inv_freq = self.inv_freq.to(
            device=q.device,
            dtype=torch.float32,
        )

        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        return (
            self._rotate_pairs(q, cos, sin),
            self._rotate_pairs(k, cos, sin),
        )

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig | AoEConfig,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        # kv_dim lưu tổng số chiều của Key_heads(Value_heads)
        kv_dim = self.head_dim * self.n_kv_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias = False)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias = False)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias = False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias = False)
        self.rope = RotaryEmbedding(self.head_dim, base = rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size() # x = input(batch_size, seq_len, d_model)
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # transpose để đưa n_heads lên trước seq_len
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k) # Áp dụng embedding xoay cho q và k

        repeat_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat_factor, dim = 1) # lặp lại k để khớp với số lượng đầu attention
        v = v.repeat_interleave(repeat_factor, dim = 1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # tính toán điểm attention

        causal_mask = torch.ones(
            seq_len,
            seq_len,
            device = x.device,
            dtype = torch.bool,
        ).triu(diagonal = 1) # tạo mặt nạ causal để ngăn chặn thông tin từ các vị trí tương lai

        scores = scores.masked_fill(causal_mask, float("-inf")) # áp dụng mặt nạ causal

        attention_weights = torch.softmax(
            scores,
            dim = -1,
            dtype = torch.float32,
        ).to(dtype = scores.dtype) # tính toán trọng số attention

        context = attention_weights @ v # tính toán ngữ cảnh bằng cách nhân trọng số attention với v
        context = context.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        ) # chuyển đổi ngữ cảnh về dạng ban đầu

        return self.o_proj(context)