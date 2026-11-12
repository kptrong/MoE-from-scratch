import math
import unittest

import torch

from moe.losses import load_balancing_loss, next_token_loss


class TestLosses(unittest.TestCase):
    def test_next_token_loss_uses_shifted_targets(self) -> None:
        logits = torch.tensor(
            [[
                [math.log(2.0), math.log(3.0), math.log(5.0)],
                [math.log(4.0), 0.0, math.log(5.0)],
                [100.0, -100.0, 0.0],
            ]],
            device="cpu",
            dtype=torch.float32,
        )
        input_ids = torch.tensor(
            [[2, 1, 0]],
            device="cpu",
            dtype=torch.long,
        )

        actual = next_token_loss(logits, input_ids)

        # Vị trí 0 dự đoán token 1, xác suất là 3/10.
        # Vị trí 1 dự đoán token 0, xác suất là 4/10.
        # Logits ở vị trí cuối không tham gia loss.
        expected = -(
            math.log(3.0 / 10.0)
            + math.log(4.0 / 10.0)
        ) / 2.0

        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_next_token_loss_rejects_short_sequences(self) -> None:
        for seq_len in (0, 1):
            with self.subTest(seq_len=seq_len):
                logits = torch.zeros(
                    1, seq_len, 3,
                    device="cpu",
                    dtype=torch.float32,
                )
                input_ids = torch.zeros(
                    1, seq_len,
                    device="cpu",
                    dtype=torch.long,
                )

                with self.assertRaisesRegex(ValueError, "at least 2"):
                    next_token_loss(logits, input_ids)

    def test_load_balancing_top_k_one(self) -> None:
        logits = torch.tensor(
            [[
                [math.log(3.0), 0.0],
                [math.log(3.0), 0.0],
            ]],
            device="cpu",
            dtype=torch.float32,
        )

        aux_loss, counts = load_balancing_loss((logits,), top_k=1)

        self.assertAlmostEqual(aux_loss.item(), 1.5, places=6)
        self.assertEqual(aux_loss.dtype, torch.float32)
        self.assertEqual(counts.dtype, torch.int64)
        self.assertFalse(counts.requires_grad)

        torch.testing.assert_close(
            counts,
            torch.tensor(
                [[2, 0]],
                device="cpu",
                dtype=torch.int64,
            ),
        )

    def test_load_balancing_top_k_two(self) -> None:
        logits = torch.tensor(
            [[[math.log(4.0), math.log(2.0), 0.0]]],
            device="cpu",
            dtype=torch.float32,
        )

        aux_loss, counts = load_balancing_loss((logits,), top_k=2)

        # P = [4/7, 2/7, 1/7], f = [1/2, 1/2, 0].
        expected = 9.0 / 7.0
        self.assertAlmostEqual(aux_loss.item(), expected, places=6)

        torch.testing.assert_close(
            counts,
            torch.tensor(
                [[1, 1, 0]],
                device="cpu",
                dtype=torch.int64,
            ),
        )

    def test_layers_are_computed_separately_then_averaged(self) -> None:
        a = math.log(4.0)
        b = math.log(2.0)

        first_layer = torch.tensor(
            [[[a, b, 0.0], [a, b, 0.0]]],
            device="cpu",
            dtype=torch.float32,
        )

        second_layer = torch.tensor(
            [[
                [a, b, 0.0],
                [0.0, a, b],
                [b, 0.0, a],
            ]],
            device="cpu",
            dtype=torch.float32,
        )

        aux_loss, counts = load_balancing_loss(
            (first_layer, second_layer),
            top_k=2,
        )

        # Lớp 0 có loss 9/7.
        # Lớp 1 có P và f đều bằng [1/3, 1/3, 1/3], loss bằng 1.
        expected = (9.0 / 7.0 + 1.0) / 2.0
        self.assertAlmostEqual(aux_loss.item(), expected, places=6)

        torch.testing.assert_close(
            counts,
            torch.tensor(
                [[2, 2, 0], [2, 2, 2]],
                device="cpu",
                dtype=torch.int64,
            ),
        )

        # Mỗi lớp có tổng số lượt chọn bằng N * top_k.
        torch.testing.assert_close(
            counts.sum(dim=-1),
            torch.tensor(
                [2 * 2, 3 * 2],
                device="cpu",
                dtype=torch.int64,
            ),
        )

    def test_aux_loss_backward(self) -> None:
        logits = torch.tensor(
            [[
                [math.log(3.0), 0.0],
                [math.log(3.0), 0.0],
            ]],
            device="cpu",
            dtype=torch.float32,
            requires_grad=True,
        )

        aux_loss, counts = load_balancing_loss((logits,), top_k=1)
        aux_loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad != 0).any().item())
        self.assertFalse(counts.requires_grad)

    def test_load_balancing_rejects_invalid_inputs(self) -> None:
        valid_logits = torch.zeros(
            1, 2, 2,
            device="cpu",
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            load_balancing_loss((), top_k=1)

        for top_k in (0, -1, 3, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    load_balancing_loss((valid_logits,), top_k=top_k)

        for shape in ((1, 0, 2), (0, 3, 2)):
            with self.subTest(shape=shape):
                empty_logits = torch.zeros(
                    shape,
                    device="cpu",
                    dtype=torch.float32,
                )

                with self.assertRaisesRegex(ValueError, "token"):
                    load_balancing_loss((empty_logits,), top_k=1)

        wrong_rank = torch.zeros(
            2, 2,
            device="cpu",
            dtype=torch.float32,
        )
        with self.assertRaises(ValueError):
            load_balancing_loss((wrong_rank,), top_k=1)

        different_expert_count = torch.zeros(
            1, 2, 3,
            device="cpu",
            dtype=torch.float32,
        )
        with self.assertRaises(ValueError):
            load_balancing_loss(
                (valid_logits, different_expert_count),
                top_k=1,
            )


if __name__ == "__main__":
    unittest.main()