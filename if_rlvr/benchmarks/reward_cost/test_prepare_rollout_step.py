from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import prepare_rollout_step as subject


class FakeBatch(dict):
    def __init__(self, values, size):
        super().__init__(values)
        self.batch_size = (size,)


class FakeProto:
    def __init__(self, batch, non_tensor_batch, size):
        self.batch = None if batch is None else FakeBatch(batch, size)
        self.non_tensor_batch = non_tensor_batch


class FakeTokenizer:
    name_or_path = "fake/Qwen3-4B"

    def __init__(self):
        self.chat_template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        return [101, 102, 103, 104, 105]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "</think>":
            return [99]
        if text == "Answer":
            return [20, 21]
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is True
        mapping = {
            (10, 99, 20, 21): "<think>work</think>\n Answer",
            (11, 99, 20, 21): "<think>other</think>\n Answer",
            (20, 21): "\n Answer",
        }
        return mapping.get(tuple(token_ids), "".join(chr(token_id) for token_id in token_ids))


def fake_remove_thinking_section(text, require_think_end=False):
    assert require_think_end is False
    return text.split("</think>", 1)[-1].strip()


def fake_score_ifeval(text, ground_truth, require_think_end=False):
    assert ground_truth == "constraint-spec"
    return 1.0 if (not require_think_end or "</think>" in text) else 0.0


class PrepareRolloutStepTest(unittest.TestCase):
    def _make_batches(self):
        prompts = [[{"role": "user", "content": "Question plus constraint"}]] * 2
        ppl_prompts = [[{"role": "user", "content": "Question"}]] * 2
        common = {
            "uid": ["uid-a", "uid-a"],
            "raw_prompt": prompts,
            "ppl_prompt": ppl_prompts,
            "reward_model": [{"ground_truth": "constraint-spec"}] * 2,
            "extra_info": [{"index": 7}, {"index": 7}],
        }
        new_batch = FakeProto(batch={"dummy": [0, 0]}, non_tensor_batch=common, size=2)
        gen_batch = FakeProto(
            batch={
                "responses": [[10, 99, 20, 21, 0], [11, 99, 20, 21, 0]],
                "response_mask": [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
            },
            non_tensor_batch={
                **common,
                "ppl_x_prompt_ids": [[301, 302], []],
                "ppl_y_final_answer_ids": [[401, 402], []],
                "if_final_answer_ids": [[], []],
            },
            size=2,
        )
        return new_batch, gen_batch

    def test_prompt_to_text_matches_reward_manager_format(self):
        self.assertEqual(subject._prompt_to_text([{"role": "user", "content": "only"}]), "only")
        self.assertEqual(
            subject._prompt_to_text(
                [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "question"},
                ]
            ),
            "system:\nrules\n\nuser:\nquestion",
        )

    def test_response_ids_are_selected_by_both_masks(self):
        gen_batch = FakeProto(
            batch={
                "responses": [[1, 2, 3, 4]],
                "response_mask": [[1, 0, 1, 1]],
                "attention_mask": [[1, 1, 1, 0]],
            },
            non_tensor_batch={},
            size=1,
        )
        self.assertEqual(subject._extract_response_token_ids(gen_batch, 0), [1, 3])

    def test_batch_size_supports_non_tensor_only_dataproto(self):
        proto = FakeProto(batch=None, non_tensor_batch={"uid": ["a", "b", "c"]}, size=3)
        self.assertEqual(subject._batch_size(proto), 3)
        self.assertFalse(subject._has_batch_key(proto, "responses"))

    def test_prefers_dumped_ppl_tokens_then_reconstructs_and_extracts_answer(self):
        new_batch, gen_batch = self._make_batches()
        tokenizer = FakeTokenizer()
        validation, uids = subject._rollout_shape(
            new_batch,
            gen_batch,
            expected_rows=2,
            expected_unique_uids=1,
            expected_rollouts_per_uid=2,
            allow_count_mismatch=False,
        )
        self.assertTrue(validation["passed"])
        rows = list(
            subject._iter_records(
                new_batch,
                gen_batch,
                uids,
                tokenizer,
                fake_remove_thinking_section,
                fake_score_ifeval,
                max_prompt_length=3,
                require_think_end=True,
                progress_every=0,
            )
        )

        self.assertEqual(rows[0]["id"], "uid-a:0")
        self.assertEqual(rows[0]["response_token_ids"], [10, 99, 20, 21])
        self.assertEqual(rows[0]["ppl_prefix_token_ids"], [301, 302])
        self.assertEqual(rows[0]["ppl_continuation_token_ids"], [401, 402])
        self.assertEqual(rows[0]["ppl_prefix_source"], "gen_batch.ppl_x_prompt_ids")
        self.assertEqual(rows[0]["ppl_continuation_source"], "gen_batch.ppl_y_final_answer_ids")
        self.assertEqual(rows[0]["judge_prompt"], "Question")
        self.assertEqual(rows[0]["judge_response"], "Answer")
        self.assertEqual(rows[0]["constraint_score"], 1.0)

        # Reconstructed prefixes are left-truncated, and y is the post-thinking answer.
        self.assertEqual(rows[1]["ppl_prefix_token_ids"], [103, 104, 105])
        self.assertEqual(rows[1]["ppl_continuation_token_ids"], [20, 21])
        self.assertEqual(rows[1]["ppl_continuation_source"], "derived:response_after_</think>")
        self.assertEqual(tokenizer.chat_template_calls[0][1]["enable_thinking"], False)
        self.assertEqual(tokenizer.chat_template_calls[0][1]["add_generation_prompt"], True)

    def test_shape_mismatch_requires_opt_in(self):
        new_batch, gen_batch = self._make_batches()
        with self.assertRaisesRegex(ValueError, "expected 8192 rows"):
            subject._rollout_shape(
                new_batch,
                gen_batch,
                expected_rows=8192,
                expected_unique_uids=1024,
                expected_rollouts_per_uid=8,
                allow_count_mismatch=False,
            )
        validation, _ = subject._rollout_shape(
            new_batch,
            gen_batch,
            expected_rows=8192,
            expected_unique_uids=1024,
            expected_rollouts_per_uid=8,
            allow_count_mismatch=True,
        )
        self.assertFalse(validation["passed"])
        self.assertTrue(validation["errors"])

    def test_empty_response_is_preserved_as_ppl_ineligible(self):
        gen_batch = FakeProto(
            batch={"dummy": [0]},
            non_tensor_batch={"ppl_y_final_answer_ids": [[]], "if_final_answer_ids": [[]]},
            size=1,
        )
        continuation, source = subject._select_ppl_continuation(
            gen_batch, FakeTokenizer(), [32, 32], 0
        )
        self.assertEqual(continuation, [])
        self.assertEqual(source, "ineligible:empty_or_whitespace_response")

    def test_readme_count_flag_aliases(self):
        args = subject._parse_args(
            [
                "genstep_000001",
                "--output",
                "rollouts.jsonl",
                "--expected-responses",
                "16",
                "--expected-prompts",
                "4",
                "--rollouts-per-prompt",
                "4",
            ]
        )
        self.assertEqual(args.expected_rows, 16)
        self.assertEqual(args.expected_unique_uids, 4)
        self.assertEqual(args.expected_rollouts_per_uid, 4)

    def test_writer_hashes_the_exact_canonical_bytes(self):
        record = {
            "response_token_count": 1,
            "ppl_prefix_token_count": 2,
            "ppl_continuation_token_count": 1,
            "constraint_score": 0.5,
            "ppl_eligible": True,
            "ppl_prefix_source": "x",
            "ppl_continuation_source": "y",
            "z": "한글",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            digest, stats = subject._write_jsonl(iter([record]), path)
            raw = path.read_bytes()
            self.assertEqual(digest, subject.hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads(raw), record)
            self.assertEqual(stats["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
