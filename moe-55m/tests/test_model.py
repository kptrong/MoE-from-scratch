from dataclasses import replace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from moe.config import (
    count_parameters,
    shared_config,
    vanilla_config,
)
from moe.model import MoELanguageModel


def small_config(shared: bool):
    base = shared_config() if shared else vanilla_config()

    return replace(
        base,
        vocab_size=32,
        n_layers=2,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        expert_hidden_dim=12,
        n_routed_experts=4,
        top_k=2,
    )


class TestMoELanguageModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_output_shapes(self) -> None:
        for shared in (False, True):
            config = small_config(shared)
            model = MoELanguageModel(config).cpu().float()

            for seq_len in (1, 5):
                with self.subTest(shared=shared, seq_len=seq_len):
                    input_ids = torch.randint(
                        config.vocab_size,
                        (2, seq_len),
                        device="cpu",
                        dtype=torch.long,
                    )

                    with torch.no_grad():
                        logits, router_logits = model(input_ids)

                    self.assertEqual(
                        logits.shape,
                        (2, seq_len, config.vocab_size),
                    )
                    self.assertTrue(torch.isfinite(logits).all().item())

                    self.assertIsInstance(router_logits, tuple)
                    self.assertEqual(len(router_logits), config.n_layers)

                    for layer_logits in router_logits:
                        self.assertEqual(
                            layer_logits.shape,
                            (2, seq_len, config.n_routed_experts),
                        )

    def test_embedding_and_head_share_parameter(self) -> None:
        for shared in (False, True):
            with self.subTest(shared=shared):
                model = MoELanguageModel(small_config(shared)).cpu()

                self.assertIs(
                    model.lm_head.weight,
                    model.token_embedding.weight,
                )
                self.assertIsNone(model.lm_head.bias)

    def test_blocks_have_independent_weights(self) -> None:
        for shared in (False, True):
            with self.subTest(shared=shared):
                config = small_config(shared)
                model = MoELanguageModel(config).cpu()

                self.assertIsInstance(model.blocks, nn.ModuleList)
                self.assertEqual(len(model.blocks), config.n_layers)

                self.assertEqual(
                    len({id(block) for block in model.blocks}),
                    config.n_layers,
                )

                parameters = [
                    parameter
                    for block in model.blocks
                    for parameter in block.parameters()
                ]

                # Không tái sử dụng cùng Parameter giữa các block.
                self.assertEqual(
                    len({id(parameter) for parameter in parameters}),
                    len(parameters),
                )

                # Không có các Parameter khác nhau dùng chung vùng nhớ.
                self.assertEqual(
                    len({parameter.data_ptr() for parameter in parameters}),
                    len(parameters),
                )

    def test_causal_prefix(self) -> None:
        prefix_len = 3

        for shared in (False, True):
            with self.subTest(shared=shared):
                config = small_config(shared)
                model = MoELanguageModel(config).cpu().float()
                model.eval()

                first_ids = torch.randint(
                    config.vocab_size,
                    (2, 7),
                    device="cpu",
                    dtype=torch.long,
                )
                second_ids = first_ids.clone()

                # Giữ nguyên prefix và thay toàn bộ token trong suffix.
                second_ids[:, prefix_len:] = (
                    second_ids[:, prefix_len:] + 1
                ) % config.vocab_size

                with torch.no_grad():
                    first_logits, _ = model(first_ids)
                    second_logits, _ = model(second_ids)

                torch.testing.assert_close(
                    first_logits[:, :prefix_len, :],
                    second_logits[:, :prefix_len, :],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_next_token_backward(self) -> None:
        for shared in (False, True):
            with self.subTest(shared=shared):
                config = small_config(shared)
                model = MoELanguageModel(config).cpu().float()

                input_ids = torch.randint(
                    config.vocab_size,
                    (2, 5),
                    device="cpu",
                    dtype=torch.long,
                )

                logits, _ = model(input_ids)

                prediction_logits = logits[:, :-1, :].reshape(
                    -1, config.vocab_size
                )
                targets = input_ids[:, 1:].reshape(-1)

                loss = F.cross_entropy(prediction_logits, targets)

                self.assertTrue(torch.isfinite(loss).item())

                loss.backward()

                shared_weight = model.token_embedding.weight

                self.assertIs(model.lm_head.weight, shared_weight)
                self.assertIsNotNone(shared_weight.grad)
                self.assertTrue(
                    torch.isfinite(shared_weight.grad).all().item()
                )
                self.assertTrue(
                    (shared_weight.grad != 0).any().item()
                )

    def test_small_parameter_counts(self) -> None:
        for shared in (False, True):
            with self.subTest(shared=shared):
                config = small_config(shared)
                model = MoELanguageModel(config).cpu()

                # parameters() chỉ đếm Parameter dùng chung một lần.
                actual = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )
                expected = count_parameters(config)["total"]

                self.assertEqual(actual, expected)

    def test_target_parameter_counts(self) -> None:
        cases = (
            ("Vanilla", vanilla_config(), 54_948_352),
            ("Shared", shared_config(), 54_944_256),
        )

        for name, config, expected in cases:
            with self.subTest(variant=name):
                theoretical = count_parameters(config)["total"]
                self.assertEqual(theoretical, expected)

                model = MoELanguageModel(config).cpu()
                actual = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )

                self.assertEqual(actual, theoretical)
                self.assertEqual(actual, expected)

                # Giải phóng mô hình lớn trước biến thể tiếp theo.
                del model


if __name__ == "__main__":
    unittest.main()