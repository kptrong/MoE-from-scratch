import math
import unittest

import torch
from torch import nn

from moe.layers import RMSNorm, SwiGLUExpert


class TestLayers(unittest.TestCase):
    def assert_finite_gradients(
        self,
        layer: nn.Module,
        x: torch.Tensor,
    ) -> None:
        self.assertIsNotNone(x.grad, "Input did not receive a gradient")
        self.assertTrue(
            torch.isfinite(x.grad).all().item(),
            "Input gradient contains NaN or infinity",
        )

        for name, parameter in layer.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(
                    parameter.grad,
                    f"{name} did not receive a gradient",
                )
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item(),
                    f"{name} gradient contains NaN or infinity",
                )

    def test_expert_preserves_shape(self) -> None:
        expert = SwiGLUExpert(d_model=16, expert_hidden_dim=12).cpu()

        for shape in ((2, 5, 16), (7, 16)):
            with self.subTest(shape=shape):
                x = torch.randn(*shape, device="cpu")
                output = expert(x)
                self.assertEqual(output.shape, x.shape)

    def test_expert_parameter_count(self) -> None:
        expert = SwiGLUExpert(d_model=512, expert_hidden_dim=384).cpu()
        count = sum(parameter.numel() for parameter in expert.parameters())
        self.assertEqual(count, 589_824)

    def test_rmsnorm_parameter_count(self) -> None:
        norm = RMSNorm(dim=512).cpu()
        count = sum(parameter.numel() for parameter in norm.parameters())
        self.assertEqual(count, 512)

    def test_rmsnorm_values_per_token(self) -> None:
        eps = 1e-5
        norm = RMSNorm(dim=2, eps=eps).cpu().double()

        torch.testing.assert_close(
            norm.weight,
            torch.ones(2, dtype=torch.float64, device="cpu"),
        )

        # Một batch gồm hai token, mỗi token có hai chiều.
        x = torch.tensor(
            [[[3.0, 4.0], [6.0, 8.0]]],
            dtype=torch.float64,
            device="cpu",
        )

        # Tính độc lập từng mẫu số bằng số vô hướng Python.
        first_denominator = math.sqrt((3.0**2 + 4.0**2) / 2.0 + eps)
        second_denominator = math.sqrt((6.0**2 + 8.0**2) / 2.0 + eps)

        expected = torch.tensor(
            [[
                [3.0 / first_denominator, 4.0 / first_denominator],
                [6.0 / second_denominator, 8.0 / second_denominator],
            ]],
            dtype=torch.float64,
            device="cpu",
        )

        output = norm(x)

        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(
            output,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_expert_backward(self) -> None:
        expert = SwiGLUExpert(d_model=16, expert_hidden_dim=12).cpu()
        x = torch.randn(2, 5, 16, device="cpu", requires_grad=True)

        loss = expert(x).square().mean()
        loss.backward()

        self.assert_finite_gradients(expert, x)

    def test_rmsnorm_backward(self) -> None:
        norm = RMSNorm(dim=16).cpu()
        x = torch.randn(2, 5, 16, device="cpu", requires_grad=True)

        loss = norm(x).square().mean()
        loss.backward()

        self.assert_finite_gradients(norm, x)


if __name__ == "__main__":
    unittest.main()