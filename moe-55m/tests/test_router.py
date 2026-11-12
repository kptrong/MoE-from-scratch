import unittest
import torch
from moe.router import TopKRouter


class TestTopKRouter(unittest.TestCase):
    def make_identity_router(self, top_k: int = 2) -> TopKRouter:
        router = TopKRouter(
            d_model=4,
            n_routed_experts=4,
            top_k=top_k,
        ).cpu()

        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(4, device="cpu"))

        return router

    def test_output_shapes_and_properties(self) -> None:
        # Dùng float64 để kiểm tra softmax vẫn trả trọng số float32.
        router = TopKRouter(
            d_model=4,
            n_routed_experts=5,
            top_k=3,
        ).cpu().double()

        for shape in ((7, 4), (2, 5, 4)):
            with self.subTest(shape=shape):
                x = torch.randn(
                    *shape,
                    dtype=torch.float64,
                    device="cpu",
                )
                indices, weights, logits = router(x)

                selected_shape = x.shape[:-1] + (3,)
                logits_shape = x.shape[:-1] + (5,)

                self.assertEqual(indices.shape, selected_shape)
                self.assertEqual(weights.shape, selected_shape)
                self.assertEqual(logits.shape, logits_shape)

                self.assertEqual(indices.dtype, torch.int64)
                self.assertTrue((indices >= 0).all().item())
                self.assertTrue((indices < 5).all().item())

                sorted_indices = indices.sort(dim=-1).values
                self.assertTrue(
                    (
                        sorted_indices[..., 1:]
                        > sorted_indices[..., :-1]
                    ).all().item()
                )

                self.assertEqual(weights.dtype, torch.float32)
                self.assertTrue(torch.isfinite(weights).all().item())
                self.assertTrue((weights >= 0).all().item())

                torch.testing.assert_close(
                    weights.sum(dim=-1),
                    torch.ones(x.shape[:-1], device="cpu"),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_identity_gate_values(self) -> None:
        router = self.make_identity_router()
        x = torch.tensor([[1.0, 3.0, 2.0, 0.0]], device="cpu")

        indices, weights, logits = router(x)

        torch.testing.assert_close(
            indices,
            torch.tensor([[1, 2]], dtype=torch.int64, device="cpu"),
        )
        torch.testing.assert_close(
            weights,
            torch.tensor(
                [[0.7310586, 0.2689414]],
                dtype=torch.float32,
                device="cpu",
            ),
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(logits, x, rtol=0, atol=0)

    def test_top_k_one(self) -> None:
        router = self.make_identity_router(top_k=1)
        x = torch.tensor([[1.0, 3.0, 2.0, 0.0]], device="cpu")

        indices, weights, _ = router(x)

        torch.testing.assert_close(
            indices,
            torch.tensor([[1]], dtype=torch.int64, device="cpu"),
        )
        torch.testing.assert_close(
            weights,
            torch.ones(1, 1, device="cpu"),
            rtol=0,
            atol=0,
        )

    def test_top_k_all_experts(self) -> None:
        router = self.make_identity_router(top_k=4)
        x = torch.tensor([[1.0, 3.0, 2.0, 0.0]], device="cpu")

        indices, weights, _ = router(x)

        # Thứ tự giảm dần theo logits.
        torch.testing.assert_close(
            indices,
            torch.tensor(
                [[1, 2, 0, 3]],
                dtype=torch.int64,
                device="cpu",
            ),
        )
        torch.testing.assert_close(
            indices.sort(dim=-1).values,
            torch.arange(4, device="cpu").unsqueeze(0),
        )
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(1, device="cpu"),
        )

    def test_backward(self) -> None:
        router = self.make_identity_router()
        x = torch.tensor(
            [[1.0, 3.0, 2.0, 0.0]],
            device="cpu",
            requires_grad=True,
        )

        _, weights, logits = router(x)

        self.assertTrue(weights.requires_grad)
        self.assertTrue(logits.requires_grad)
        logits.retain_grad()

        coefficients = torch.tensor([1.0, 2.0], device="cpu")
        loss = (weights * coefficients).sum()
        loss.backward()

        gradients = {
            "input": x.grad,
            "gate": router.gate.weight.grad,
            "logits": logits.grad,
        }
        for name, gradient in gradients.items():
            with self.subTest(gradient=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertTrue((gradient != 0).any().item())

    def test_parameter_counts(self) -> None:
        cases = (
            ("Vanilla", 512, 9, 3, 4_608),
            ("Shared", 512, 8, 2, 4_096),
        )
        for name, d_model, n_experts, top_k, expected in cases:
            with self.subTest(router=name):
                router = TopKRouter(d_model, n_experts, top_k).cpu()
                count = sum(
                    parameter.numel()
                    for parameter in router.parameters()
                )
                self.assertEqual(count, expected)

    def test_arguments_must_be_positive_integers(self) -> None:
        valid = {
            "d_model": 4,
            "n_routed_experts": 4,
            "top_k": 2,
        }

        for name in valid:
            for invalid_value in (0, -1, 1.5, True):
                with self.subTest(
                    argument=name,
                    value=invalid_value,
                ):
                    arguments = valid.copy()
                    arguments[name] = invalid_value

                    with self.assertRaisesRegex(ValueError, name):
                        TopKRouter(**arguments)

    def test_top_k_exceeds_expert_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            TopKRouter(d_model=4, n_routed_experts=4, top_k=5)


if __name__ == "__main__":
    unittest.main()