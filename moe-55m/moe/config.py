from dataclasses import dataclass, replace

# dataclasses: Một mô-đun trong Python cung cấp một cách tiện lợi để định nghĩa các lớp dữ liệu (data classes) mà không cần phải viết nhiều mã lặp lại. Nó tự động tạo ra các phương thức như __init__, __repr__, __eq__, và các phương thức khác dựa trên các trường được định nghĩa trong lớp.

@dataclass(frozen = True) # Một lớp dữ liệu bất biến (immutable) để lưu trữ cấu hình của mô hình
# class ModelConfig: Lớp này định nghĩa cấu hình của mô hình, bao gồm các tham số như kích thước từ vựng, số lượng lớp, kích thước mô hình, số lượng đầu attention, và các thông số liên quan đến các chuyên gia (experts).
class ModelConfig:
    vocab_size: int # The size of the vocabulary
    n_layers: int # The number of layers
    d_model: int # The dimension of the model
    n_heads: int # The number of attention heads
    n_kv_heads: int # The number of key-value attention heads
    expert_hidden_dim: int # The hidden dimension of the experts
    n_routed_experts: int # The number of routed experts
    n_shared_experts: int # The number of shared experts
    top_k: int # The top-k experts to select

    def __post_init__(self) -> None:

        # Một tuple chứa tên các trường phải là số nguyên dương
        positive_fields = (
            "vocab_size",
            "n_layers",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "expert_hidden_dim",
            "n_routed_experts",
        )
        for name in positive_fields:
            value = getattr(self, name) # Lấy giá trị của trường hiện tại
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value}")

        # Kiểm tra các trường phải là số nguyên không âm
        if type(self.n_shared_experts) is not int or self.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be a non-negative integer")

        # Kiểm tra top_k phải là số nguyên trong khoảng từ 1 đến n_routed_experts
        if type(self.top_k) is not int or not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError("top_k must be an integer in [1, n_routed_experts]")

        # Kiểm tra tính hợp lệ của các trường liên quan đến attention heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim (d_model // n_heads) must be even")


@dataclass(frozen=True)
class UoEConfig(ModelConfig):
    n_routing_neurons: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_shared_experts != 0:
            raise ValueError("UoE requires n_shared_experts=0")
        if (
            type(self.n_routing_neurons) is not int
            or not 1 <= self.n_routing_neurons < self.expert_hidden_dim
        ):
            raise ValueError(
                "n_routing_neurons must be an integer in [1, expert_hidden_dim)"
            )


def vanilla_config() -> ModelConfig:
    return ModelConfig(
        vocab_size = 12000,
        n_layers = 8,
        d_model = 512,
        n_heads = 8,
        n_kv_heads = 4,
        expert_hidden_dim = 384,
        n_routed_experts = 9,
        n_shared_experts = 0,
        top_k = 3,        
    )

def shared_config() -> ModelConfig:
    return replace(vanilla_config(), n_routed_experts = 8, n_shared_experts = 1, top_k = 2)


def uoe_config() -> UoEConfig:
    shared = shared_config()
    return UoEConfig(
        vocab_size=shared.vocab_size,
        n_layers=shared.n_layers,
        d_model=shared.d_model,
        n_heads=shared.n_heads,
        n_kv_heads=shared.n_kv_heads,
        expert_hidden_dim=shared.expert_hidden_dim,
        n_routed_experts=shared.n_routed_experts,
        n_shared_experts=0,
        top_k=shared.top_k,
        n_routing_neurons=round(shared.expert_hidden_dim / shared.n_routed_experts),
    )


def _count_backbone_parameters(config: ModelConfig) -> tuple[int, int, int]:
    d_model = config.d_model
    n_layers = config.n_layers
    head_dim = d_model // config.n_heads
    kv_dim = config.n_kv_heads * head_dim

    embedding_head = config.vocab_size * d_model # Tính toán số lượng tham số trong lớp embedding: vocab_size * d_model
    attention = n_layers * (2 * d_model * d_model + 2 * d_model * kv_dim) # Tính toán số lượng tham số trong các lớp attention
    rmsnorm = (2 * n_layers + 1) * d_model
    return embedding_head, attention, rmsnorm


def count_parameters(config: ModelConfig) -> dict[str, int]:
    if isinstance(config, UoEConfig):
        return count_uoe_parameters(config)
    d_model = config.d_model
    n_layers = config.n_layers
    embedding_head, attention, rmsnorm = _count_backbone_parameters(config)
    router = n_layers * d_model * config.n_routed_experts

    per_expert = 3 * d_model * config.expert_hidden_dim
    expert_total = n_layers * per_expert * (config.n_routed_experts + config.n_shared_experts)
    expert_active = n_layers * per_expert * (config.top_k + config.n_shared_experts)
    base = embedding_head + attention + rmsnorm + router

    return {
        "embedding_head": embedding_head,
        "attention": attention,
        "rmsnorm": rmsnorm,
        "router": router,
        "expert_total": expert_total,
        "expert_active": expert_active,
        "total": base + expert_total,
        "active": base + expert_active,
    }


def count_uoe_parameters(config: UoEConfig) -> dict[str, int]:
    embedding_head, attention, rmsnorm = _count_backbone_parameters(config)
    backbone = embedding_head + attention + rmsnorm
    weights_per_neuron = 3 * config.d_model

    expert_total = (
        config.n_layers
        * weights_per_neuron
        * config.n_routed_experts
        * config.expert_hidden_dim
    )
    # Selected experts already include their routing neurons.
    active_neurons = (
        config.top_k * config.expert_hidden_dim
        + (config.n_routed_experts - config.top_k) * config.n_routing_neurons
    )
    expert_active = config.n_layers * weights_per_neuron * active_neurons

    return {
        "embedding_head": embedding_head,
        "attention": attention,
        "rmsnorm": rmsnorm,
        "router": 0,
        "expert_total": expert_total,
        "expert_active": expert_active,
        "total": backbone + expert_total,
        "active": backbone + expert_active,
    }


if __name__ == "__main__":
    uoe = uoe_config()
    print(f"{'Preset':<8} {'total':>12} {'active':>12}")
    for name, counts, expected_total, expected_active in (
        ("Vanilla", count_parameters(vanilla_config()), 54_948_352, 26_636_800),
        ("Shared", count_parameters(shared_config()), 54_944_256, 26_632_704),
        ("UoE", count_uoe_parameters(uoe), 50_192_896, 25_420_288),
    ):
        assert counts["total"] == expected_total, (name, counts)
        assert counts["active"] == expected_active, (name, counts)
        print(f"{name:<8} {counts['total']:>12,} {counts['active']:>12,}")

    expert_params = count_uoe_parameters(replace(uoe, n_layers=1))["expert_total"]
    assert expert_params == 4_718_592, expert_params
    print(f"UoE expert parameters per layer: {expert_params:,}")

    for field_name, value in (
        ("n_routing_neurons", 384),
        ("top_k", 10),
        ("n_shared_experts", 1),
    ):
        try:
            replace(uoe, **{field_name: value})
        except ValueError as error:
            assert field_name in str(error), error
            print(f"Invalid UoE {field_name}={value}: {error}")
        else:
            raise AssertionError(f"Expected ValueError for {field_name}={value}")

    small = ModelConfig(
        vocab_size=32,
        n_layers=2,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        expert_hidden_dim=12,
        n_routed_experts=3,
        n_shared_experts=2,
        top_k=1,
    )
    counts = count_parameters(small)
    assert counts == {
        "embedding_head": 512,
        "attention": 1_536,
        "rmsnorm": 80,
        "router": 96,
        "expert_total": 5_760,
        "expert_active": 3_456,
        "total": 7_984,
        "active": 5_680,
    }, counts
    print(f"Small: total={counts['total']:_}, active={counts['active']:_}")

    try:
        replace(small, top_k=4)
    except ValueError as error:
        assert "top_k" in str(error), error
        print(f"Invalid config: {error}")
    else:
        raise AssertionError("Expected ValueError for invalid top_k")
