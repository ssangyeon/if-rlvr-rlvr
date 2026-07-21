from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from flops import ModelSpec, estimate_generation, estimate_ppl  # noqa: E402


QWEN3_4B = {
    "model_type": "qwen3",
    "vocab_size": 151936,
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "tie_word_embeddings": True,
}

QWEN3_30B_A3B = {
    "model_type": "qwen3_moe",
    "vocab_size": 151936,
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "mlp_only_layers": [],
    "decoder_sparse_step": 1,
    "tie_word_embeddings": False,
}

GPT_OSS_120B = {
    "model_type": "gpt_oss",
    "vocab_size": 201088,
    "hidden_size": 2880,
    "num_hidden_layers": 36,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "intermediate_size": 2880,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "sliding_window": 128,
    "layer_types": [
        "sliding_attention" if index % 2 == 0 else "full_attention"
        for index in range(36)
    ],
    "tie_word_embeddings": False,
}


class ModelSpecTest(unittest.TestCase):
    def test_qwen3_4b_constants(self) -> None:
        spec = ModelSpec.from_config(QWEN3_4B)
        summary = spec.to_dict()

        self.assertEqual(summary["num_dense_layers"], 36)
        self.assertEqual(summary["num_moe_layers"], 0)
        self.assertEqual(summary["num_full_attention_layers"], 36)
        self.assertEqual(summary["num_sliding_attention_layers"], 0)
        self.assertEqual(
            summary["attention_projection_flops_per_token_per_layer"], 52_428_800
        )
        self.assertEqual(summary["dense_mlp_flops_per_token_per_layer"], 149_422_080)
        self.assertEqual(summary["linear_flops_per_processed_token"], 7_266_631_680)
        self.assertEqual(summary["attention_flops_per_pair_per_layer"], 16_384)
        self.assertEqual(summary["lm_head_flops_per_position"], 777_912_320)

        one_token = estimate_ppl(spec, 1)
        self.assertEqual(
            one_token,
            {
                "linear": 7_266_631_680,
                "attention": 589_824,
                "lm_head": 777_912_320,
                "total": 8_045_133_824,
            },
        )

    def test_qwen3_30b_a3b_active_moe_constants(self) -> None:
        spec = ModelSpec.from_config(QWEN3_30B_A3B)
        summary = spec.to_dict()

        self.assertEqual(summary["num_dense_layers"], 0)
        self.assertEqual(summary["num_moe_layers"], 48)
        self.assertEqual(
            summary["attention_projection_flops_per_token_per_layer"], 37_748_736
        )
        self.assertEqual(summary["moe_mlp_flops_per_token_per_layer"], 76_021_760)
        self.assertEqual(summary["linear_flops_per_processed_token"], 5_460_983_808)
        self.assertEqual(summary["attention_flops_per_pair_per_layer"], 16_384)
        self.assertEqual(summary["lm_head_flops_per_position"], 622_329_856)

    def test_gpt_oss_constants_and_sliding_boundary(self) -> None:
        spec = ModelSpec.from_config(GPT_OSS_120B)
        summary = spec.to_dict()

        self.assertEqual(summary["num_moe_layers"], 36)
        self.assertEqual(summary["num_full_attention_layers"], 18)
        self.assertEqual(summary["num_sliding_attention_layers"], 18)
        self.assertEqual(summary["sliding_window"], 128)
        self.assertEqual(
            summary["attention_projection_flops_per_token_per_layer"], 53_084_160
        )
        self.assertEqual(summary["moe_mlp_flops_per_token_per_layer"], 199_802_880)
        self.assertEqual(summary["linear_flops_per_processed_token"], 9_103_933_440)
        self.assertEqual(summary["attention_flops_per_pair_per_layer"], 16_384)
        self.assertEqual(summary["lm_head_flops_per_position"], 1_158_266_880)

        at_window = estimate_ppl(spec, 128)
        full_only_at_window = 16_384 * 36 * (128 * 129 // 2)
        self.assertEqual(at_window["attention"], full_only_at_window)

        past_window = estimate_ppl(spec, 129)
        full_pairs = 129 * 130 // 2
        sliding_pairs = 128 * 129 // 2 + 128
        expected = 16_384 * (18 * full_pairs + 18 * sliding_pairs)
        self.assertEqual(past_window["attention"], expected)
        self.assertLess(past_window["attention"], 16_384 * 36 * full_pairs)

    def test_gpt_oss_infers_alternating_layers_when_layer_types_absent(self) -> None:
        config = dict(GPT_OSS_120B)
        del config["layer_types"]
        spec = ModelSpec.from_config(config)
        self.assertEqual(spec.num_sliding_attention_layers, 18)
        self.assertEqual(spec.num_full_attention_layers, 18)


class EstimateBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ModelSpec.from_config(QWEN3_4B)

    def test_zero_length_ppl_is_zero(self) -> None:
        self.assertEqual(
            estimate_ppl(self.spec, 0),
            {"linear": 0, "attention": 0, "lm_head": 0, "total": 0},
        )

    def test_zero_vs_one_completion_token(self) -> None:
        prompt_tokens = 5
        zero = estimate_generation(self.spec, prompt_tokens, 0)
        one = estimate_generation(self.spec, prompt_tokens, 1)

        expected_linear = prompt_tokens * self.spec.linear_flops_per_processed_token
        expected_attention = (
            self.spec.attention_flops_per_pair_per_layer
            * self.spec.num_hidden_layers
            * (prompt_tokens * (prompt_tokens + 1) // 2)
        )
        self.assertEqual(zero["linear"], expected_linear)
        self.assertEqual(zero["attention"], expected_attention)
        self.assertEqual(zero["lm_head"], 0)
        self.assertEqual(one["linear"], zero["linear"])
        self.assertEqual(one["attention"], zero["attention"])
        self.assertEqual(one["lm_head"], self.spec.lm_head_flops_per_position)
        self.assertEqual(one["total"] - zero["total"], self.spec.lm_head_flops_per_position)

    def test_second_completion_token_adds_one_decode_forward(self) -> None:
        one = estimate_generation(self.spec, 5, 1)
        two = estimate_generation(self.spec, 5, 2)

        self.assertEqual(
            two["linear"] - one["linear"], self.spec.linear_flops_per_processed_token
        )
        # The new decode query attends to six keys in every full-attention layer.
        self.assertEqual(
            two["attention"] - one["attention"],
            self.spec.attention_flops_per_pair_per_layer
            * self.spec.num_hidden_layers
            * 6,
        )
        self.assertEqual(
            two["lm_head"] - one["lm_head"], self.spec.lm_head_flops_per_position
        )

    def test_cached_prefix_charges_only_new_queries(self) -> None:
        result = estimate_generation(
            self.spec, prompt_tokens=5, completion_tokens=1, cached_prompt_tokens=3
        )
        self.assertEqual(result["linear"], 2 * self.spec.linear_flops_per_processed_token)
        self.assertEqual(
            result["attention"],
            self.spec.attention_flops_per_pair_per_layer
            * self.spec.num_hidden_layers
            * ((5 * 6 // 2) - (3 * 4 // 2)),
        )
        self.assertEqual(result["lm_head"], self.spec.lm_head_flops_per_position)

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_ppl(self.spec, -1)
        with self.assertRaises(ValueError):
            estimate_generation(self.spec, 5, -1)
        with self.assertRaises(ValueError):
            estimate_generation(self.spec, 5, 1, cached_prompt_tokens=6)


if __name__ == "__main__":
    unittest.main()

