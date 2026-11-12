from dataclasses import replace
import unittest

import torch
from torch import nn

from moe.config import ModelConfig, shared_config, vanilla_config
from moe.moe_layer import MoELayer


def small_config(shared: bool) -> ModelConfig:
    base = shared_config() if shared else vanilla_config()

    return replace(
        base,
        vocab_size=32,
        n_layers=1,
        d_model=4,
        n_heads=2,
        n_kv_heads=1,
        expert_hidden_dim=3,
        n_routed_experts=4,
        top_k=2,
    )


def token_reference(
    layer: MoELayer,
    x: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tính độc lập từng token, không gọi router.forward."""
    flat_x = x.reshape(-1, x.shape[-1])
    outputs = []
    logits_per_token = []

    for token in flat_x:
        logits = layer.router.gate.weight @ token
        logits_per_token.append(logits)

        selected_logits, selected_indices = torch.topk(
            logits,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        weights = torch.softmax(
            selected_logits,
            dim=-1,
            dtype=torch.float32,
        )

        token_output = torch.zeros_like(token)

        for position, expert_id in enumerate(selected_indices.tolist()):
            expert_output = layer.routed_experts[expert_id](token)
            weight = weights[position].to(dtype=expert_output.dtype)
            token_output = token_output + weight * expert_output

        for expert in layer.shared_experts:
            token_output = token_output + expert(token)

        outputs.append(token_output)

    output = torch.stack(outputs).reshape(x.shape)
    logits = torch.stack(logits_per_token)
    logits = logits.reshape(x.shape[:-1] + (logits.shape[-1],))

    return output, logits


class ConstantExpert(nn.Module):
    """Expert dùng riêng trong test, trả cùng một vector cho mọi token."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(x)


class TestMoELayer(unittest.TestCase):
    def make_fixed_case(self):
        config = small_config(shared=True)
        layer = MoELayer(config).cpu()

        with torch.no_grad():
            # Với x[..., 0] > 0, thứ tự expert luôn là 0, 1, 2, 3.
            layer.router.gate.weight.copy_(
                torch.tensor(
                    [
                        [4.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                    device="cpu",
                )
            )

            # Các expert có đầu ra khác nhau và có thể tái lập.
            experts = list(layer.routed_experts) + list(layer.shared_experts)
            for expert_id, expert in enumerate(experts):
                for parameter in expert.parameters():
                    parameter.fill_(0.1 * (expert_id + 1))

        x = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [2.0, -0.5, 1.0, 1.0],
                [3.0, 1.0, 0.25, -1.0],
            ],
            device="cpu",
        )

        return layer, x, config

    def assert_finite_gradient(self, tensor: torch.Tensor) -> None:
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all().item())

    def test_shapes_and_token_reference(self) -> None:
        torch.manual_seed(0)

        for shared in (False, True):
            config = small_config(shared)
            layer = MoELayer(config).cpu()

            for shape in ((5, 4), (2, 3, 4)):
                with self.subTest(shared=shared, shape=shape):
                    x = torch.randn(*shape, device="cpu")

                    with torch.no_grad():
                        output, logits = layer(x)
                        expected_output, expected_logits = token_reference(
                            layer, x, config.top_k
                        )

                    self.assertEqual(output.shape, x.shape)
                    self.assertEqual(
                        logits.shape,
                        x.shape[:-1] + (config.n_routed_experts,),
                    )

                    torch.testing.assert_close(
                        output,
                        expected_output,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                    torch.testing.assert_close(
                        logits,
                        expected_logits,
                        rtol=1e-5,
                        atol=1e-6,
                    )

    def test_shared_experts_have_coefficient_one(self) -> None:
        config = replace(
            small_config(shared=True),
            n_shared_experts=2,
        )
        layer = MoELayer(config).cpu()

        # Loại đóng góp routed để kiểm tra riêng tổng shared.
        with torch.no_grad():
            for expert in layer.routed_experts:
                for parameter in expert.parameters():
                    parameter.zero_()

        first = torch.tensor([1.0, -2.0, 3.0, 0.5], device="cpu")
        second = torch.tensor([-0.25, 1.0, 2.0, 0.5], device="cpu")

        layer.shared_experts = nn.ModuleList([
            ConstantExpert(first),
            ConstantExpert(second),
        ])

        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.0, 3.0]],
                [[0.0, 0.0, 0.0, 0.0], [5.0, -1.0, 2.0, 1.0]],
            ],
            device="cpu",
        )

        with torch.no_grad():
            output, _ = layer(x)

        expected = (first + second).expand_as(x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_unselected_experts_receive_no_tokens(self) -> None:
        layer, x, config = self.make_fixed_case()
        token_counts = [0] * config.n_routed_experts
        handles = []

        for expert_id, expert in enumerate(layer.routed_experts):
            def record_tokens(
                module,
                inputs,
                output,
                expert_id=expert_id,
            ):
                token_counts[expert_id] += inputs[0].shape[0]

            handles.append(expert.register_forward_hook(record_tokens))

        try:
            with torch.no_grad():
                output, logits = layer(x)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(token_counts, [3, 3, 0, 0])

        expected_logits = x[:, :1] * torch.tensor(
            [4.0, 3.0, 2.0, 1.0],
            device="cpu",
        )
        torch.testing.assert_close(logits, expected_logits)

        with torch.no_grad():
            expected_output, _ = token_reference(layer, x, config.top_k)

        torch.testing.assert_close(
            output,
            expected_output,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_backward(self) -> None:
        layer, x, _ = self.make_fixed_case()
        x.requires_grad_()

        output, _ = layer(x)
        loss = output.square().mean()
        loss.backward()

        self.assert_finite_gradient(x)

        # Gate cố định và đầu vào dương ở chiều đầu bảo đảm chọn expert 0, 1.
        participating_modules = [
            ("router", layer.router),
            ("routed_0", layer.routed_experts[0]),
            ("routed_1", layer.routed_experts[1]),
        ]
        participating_modules.extend(
            (f"shared_{index}", expert)
            for index, expert in enumerate(layer.shared_experts)
        )

        for module_name, module in participating_modules:
            for parameter_name, parameter in module.named_parameters():
                with self.subTest(
                    module=module_name,
                    parameter=parameter_name,
                ):
                    self.assert_finite_gradient(parameter)

        self.assertTrue(
            (layer.router.gate.weight.grad != 0).any().item()
        )

    def test_experts_have_independent_weights(self) -> None:
        for shared in (False, True):
            with self.subTest(shared=shared):
                config = small_config(shared)
                layer = MoELayer(config).cpu()

                self.assertIsInstance(layer.routed_experts, nn.ModuleList)
                self.assertIsInstance(layer.shared_experts, nn.ModuleList)
                self.assertEqual(
                    len(layer.routed_experts),
                    config.n_routed_experts,
                )
                self.assertEqual(
                    len(layer.shared_experts),
                    config.n_shared_experts,
                )

                experts = (
                    list(layer.routed_experts)
                    + list(layer.shared_experts)
                )

                self.assertEqual(
                    len({id(expert) for expert in experts}),
                    len(experts),
                )

                parameters = [
                    parameter
                    for expert in experts
                    for parameter in expert.parameters()
                ]

                self.assertEqual(
                    len({id(parameter) for parameter in parameters}),
                    len(parameters),
                )
                self.assertEqual(
                    len({parameter.data_ptr() for parameter in parameters}),
                    len(parameters),
                )

    def test_parameter_counts(self) -> None:
        cases = (
            ("Vanilla", vanilla_config(), 5_313_024),
            ("Shared", shared_config(), 5_312_512),
        )

        for name, config, expected in cases:
            with self.subTest(variant=name):
                layer = MoELayer(config).cpu()
                count = sum(
                    parameter.numel()
                    for parameter in layer.parameters()
                )
                self.assertEqual(count, expected)


if __name__ == "__main__":
    unittest.main()