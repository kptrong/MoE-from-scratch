from dataclasses import replace
import math
import unittest

import torch
from torch.nn import functional as F

from moe.attention import CausalSelfAttention, RotaryEmbedding
from moe.config import vanilla_config


def small_config():
    return replace(
        vanilla_config(),
        vocab_size=32,
        n_layers=1,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        expert_hidden_dim=8,
        n_routed_experts=3,
        n_shared_experts=0,
        top_k=2,
    )


def sdpa_reference(
    attention: CausalSelfAttention,
    x: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    q = attention.q_proj(x).reshape(
        batch_size,
        seq_len,
        attention.n_heads,
        attention.head_dim,
    ).transpose(1, 2)

    k = attention.k_proj(x).reshape(
        batch_size,
        seq_len,
        attention.n_kv_heads,
        attention.head_dim,
    ).transpose(1, 2)

    v = attention.v_proj(x).reshape(
        batch_size,
        seq_len,
        attention.n_kv_heads,
        attention.head_dim,
    ).transpose(1, 2)

    q, k = attention.rope(q, k)

    repeat_factor = attention.n_heads // attention.n_kv_heads
    k = k.repeat_interleave(repeat_factor, dim=1)
    v = v.repeat_interleave(repeat_factor, dim=1)

    context = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        dropout_p=0.0,
    )

    context = context.transpose(1, 2).reshape(
        batch_size, seq_len, attention.d_model
    )
    return attention.o_proj(context)


class TestAttention(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_attention_shapes(self) -> None:
        attention = CausalSelfAttention(small_config()).cpu().float()

        for seq_len in (1, 2, 5, 9):
            with self.subTest(seq_len=seq_len):
                x = torch.randn(
                    2, seq_len, 16,
                    device="cpu",
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    output = attention(x)

                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all().item())

    def test_rope_position_zero_is_unchanged(self) -> None:
        rope = RotaryEmbedding(head_dim=4).cpu()

        q = torch.randn(2, 4, 5, 4, device="cpu", dtype=torch.float32)
        k = torch.randn(2, 2, 5, 4, device="cpu", dtype=torch.float32)

        rotated_q, rotated_k = rope(q, k)

        self.assertEqual(rotated_q.shape, q.shape)
        self.assertEqual(rotated_k.shape, k.shape)

        torch.testing.assert_close(
            rotated_q[:, :, 0, :],
            q[:, :, 0, :],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rotated_k[:, :, 0, :],
            k[:, :, 0, :],
            rtol=0,
            atol=0,
        )

    def test_rope_position_one_uses_adjacent_pairs(self) -> None:
        rope = RotaryEmbedding(head_dim=4, base=10_000.0).cpu()

        q = torch.arange(
            1, 17, device="cpu", dtype=torch.float32
        ).reshape(1, 2, 2, 4)

        k = torch.arange(
            21, 29, device="cpu", dtype=torch.float32
        ).reshape(1, 1, 2, 4)

        rotated_q, rotated_k = rope(q, k)

        # Tính kỳ vọng độc lập bằng math, với góc 1 và 0.01 radian.
        def expected_rotation(values):
            a, b, c, d = values
            return [
                a * math.cos(1.0) - b * math.sin(1.0),
                a * math.sin(1.0) + b * math.cos(1.0),
                c * math.cos(0.01) - d * math.sin(0.01),
                c * math.sin(0.01) + d * math.cos(0.01),
            ]

        expected_q = torch.tensor(
            [[
                expected_rotation(values)
                for values in q[0, :, 1, :].tolist()
            ]],
            device="cpu",
            dtype=torch.float32,
        )
        expected_k = torch.tensor(
            [[
                expected_rotation(values)
                for values in k[0, :, 1, :].tolist()
            ]],
            device="cpu",
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            rotated_q[:, :, 1, :],
            expected_q,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            rotated_k[:, :, 1, :],
            expected_k,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_rope_preserves_squared_norm(self) -> None:
        rope = RotaryEmbedding(head_dim=8).cpu()

        q = torch.randn(2, 4, 7, 8, device="cpu", dtype=torch.float32)
        k = torch.randn(2, 2, 7, 8, device="cpu", dtype=torch.float32)

        rotated_q, rotated_k = rope(q, k)

        for original, rotated in ((q, rotated_q), (k, rotated_k)):
            torch.testing.assert_close(
                rotated.square().sum(dim=-1),
                original.square().sum(dim=-1),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_rope_rejects_invalid_arguments(self) -> None:
        for head_dim in (0, -2, 3, 4.0, True):
            with self.subTest(head_dim=head_dim):
                with self.assertRaisesRegex(ValueError, "head_dim"):
                    RotaryEmbedding(head_dim)

        for base in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(base=base):
                with self.assertRaisesRegex(ValueError, "base"):
                    RotaryEmbedding(4, base=base)

    def test_kv_heads_repeat_in_contiguous_groups(self) -> None:
        attention = CausalSelfAttention(small_config()).cpu().float()

        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.k_proj.weight.zero_()
            attention.v_proj.weight.zero_()
            attention.o_proj.weight.copy_(
                torch.eye(16, device="cpu", dtype=torch.float32)
            )

            # Mọi Q head đều nhận vector [1, 0, 0, 0] ở cả hai token.
            for head in range(4):
                attention.q_proj.weight[head * 4, :2] = 1.0

            # Token 0 có K bằng 0.
            # Token 1 có K head 0 = [2, 0, 0, 0],
            #                K head 1 = [-2, 0, 0, 0].
            attention.k_proj.weight[0, 1] = 2.0
            attention.k_proj.weight[4, 1] = -2.0

            # Token 0 có V bằng 0.
            # Token 1 có hai V head [1,2,3,4] và [5,6,7,8].
            attention.v_proj.weight[:, 1] = torch.arange(
                1, 9, device="cpu", dtype=torch.float32
            )

        x = torch.zeros(1, 2, 16, device="cpu", dtype=torch.float32)
        x[0, 0, 0] = 1.0
        x[0, 1, 1] = 1.0

        with torch.no_grad():
            output = attention(x)

        # Ở token 1, scores là [0, 1] cho nhóm KV 0,
        # và [0, -1] cho nhóm KV 1.
        p = math.exp(1.0) / (1.0 + math.exp(1.0))
        q = 1.0 / (1.0 + math.exp(1.0))

        first_head = p * torch.tensor(
            [1.0, 2.0, 3.0, 4.0], device="cpu"
        )
        second_head = q * torch.tensor(
            [5.0, 6.0, 7.0, 8.0], device="cpu"
        )

        # Q head 0,1 dùng KV 0. Q head 2,3 dùng KV 1.
        expected = torch.cat([
            first_head, first_head, second_head, second_head
        ])

        torch.testing.assert_close(
            output[0, 0],
            torch.zeros(16, device="cpu"),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output[0, 1],
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_causal_prefix_is_independent_of_suffix(self) -> None:
        attention = CausalSelfAttention(small_config()).cpu().float()
        prefix_len = 3

        first = torch.randn(
            2, 7, 16, device="cpu", dtype=torch.float32
        )
        second = first.clone()
        second[:, prefix_len:, :] += 10.0 * torch.randn_like(
            second[:, prefix_len:, :]
        )

        with torch.no_grad():
            first_output = attention(first)
            second_output = attention(second)

        torch.testing.assert_close(
            first_output[:, :prefix_len, :],
            second_output[:, :prefix_len, :],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_attention_matches_sdpa_reference(self) -> None:
        attention = CausalSelfAttention(small_config()).cpu().float()

        for seq_len in (1, 3, 7):
            with self.subTest(seq_len=seq_len):
                x = torch.randn(
                    2, seq_len, 16,
                    device="cpu",
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    actual = attention(x)
                    expected = sdpa_reference(attention, x)

                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_backward(self) -> None:
        attention = CausalSelfAttention(small_config()).cpu().float()
        x = torch.randn(
            2, 5, 16,
            device="cpu",
            dtype=torch.float32,
            requires_grad=True,
        )

        loss = attention(x).square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            with self.subTest(projection=name):
                projection = getattr(attention, name)
                self.assertIsNone(projection.bias)
                self.assertIsNotNone(projection.weight.grad)
                self.assertTrue(
                    torch.isfinite(projection.weight.grad).all().item()
                )

    def test_parameter_counts(self) -> None:
        attention = CausalSelfAttention(vanilla_config()).cpu().float()

        attention_count = sum(
            parameter.numel()
            for parameter in attention.parameters()
        )
        rope_count = sum(
            parameter.numel()
            for parameter in attention.rope.parameters()
        )

        self.assertEqual(attention_count, 786_432)
        self.assertEqual(rope_count, 0)
        self.assertIn("inv_freq", dict(attention.rope.named_buffers()))


if __name__ == "__main__":
    unittest.main()