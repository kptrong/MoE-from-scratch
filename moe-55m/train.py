import argparse
from dataclasses import replace

import torch
from torch.nn.utils import clip_grad_norm_

from moe.config import shared_config, vanilla_config
from moe.losses import load_balancing_loss, next_token_loss
from moe.model import MoELanguageModel


@torch.no_grad()
def measure_lm_loss(
    model: MoELanguageModel,
    input_ids: torch.Tensor,
) -> float:
    logits, _ = model(input_ids)
    loss = next_token_loss(logits, input_ids)

    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite LM loss during evaluation")

    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small MoE model on one fixed random batch"
    )
    parser.add_argument(
        "--variant",
        choices=("vanilla", "shared"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")

    torch.manual_seed(args.seed)

    base_config = (
        vanilla_config()
        if args.variant == "vanilla"
        else shared_config()
    )
    config = replace(
        base_config,
        vocab_size=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        expert_hidden_dim=24,
    )

    # Tạo trước model để cùng seed cho cùng batch ở cả hai biến thể.
    input_ids = torch.randint(
        config.vocab_size,
        (2, 16),
        device="cpu",
        dtype=torch.long,
    )

    model = MoELanguageModel(config).to(
        device="cpu",
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.01,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Variant = {args.variant}")
    print(f"Config = {config}")
    print(f"Parameters = {parameter_count:_}")

    model.eval()
    initial_lm_loss = measure_lm_loss(model, input_ids)

    model.train()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        logits, router_logits_per_layer = model(input_ids)
        lm_loss = next_token_loss(logits, input_ids)

        # Chỉ thống kê những vị trí có tham gia next-token loss.
        router_logits_for_loss = tuple(
            layer_logits[:, :-1, :]
            for layer_logits in router_logits_per_layer
        )
        aux_loss, expert_counts = load_balancing_loss(
            router_logits_for_loss,
            top_k=config.top_k,
        )

        total_loss = lm_loss + 0.01 * aux_loss

        if not torch.isfinite(total_loss).item():
            raise RuntimeError(
                f"Non-finite loss at step {step}. "
                f"LM={lm_loss.item()}, "
                f"aux={aux_loss.item()}, "
                f"total={total_loss.item()}"
            )

        total_loss.backward()

        clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        if step % 20 == 0:
            print(
                f"Step {step:4d} | "
                f"LM={lm_loss.item():.6f} | "
                f"aux={aux_loss.item():.6f} | "
                f"total={total_loss.item():.6f}"
            )

            for layer_index, counts in enumerate(expert_counts.tolist()):
                print(f"  Layer {layer_index} expert_counts={counts}")

    # Đo lại sau lần optimizer.step() cuối cùng.
    model.eval()
    final_lm_loss = measure_lm_loss(model, input_ids)

    print(f"LM loss before training = {initial_lm_loss:.6f}")
    print(f"LM loss after training  = {final_lm_loss:.6f}")


if __name__ == "__main__":
    main()