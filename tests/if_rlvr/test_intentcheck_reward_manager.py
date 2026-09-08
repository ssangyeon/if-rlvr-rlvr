"""CPU-only behavior tests: mock network/VERL, execute the real reward manager."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[2]


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class FakeBase:
    def __init__(self, config, tokenizer, compute_score=None):
        self.config = config
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()


def load_manager():
    # Keep the test runnable without loading GPU/distributed libraries. These
    # are only import boundaries; _judge and run_single themselves are real.
    modules = {}
    for name in ("aiohttp", "verl", "verl.experimental", "verl.experimental.reward_loop",
                 "verl.experimental.reward_loop.reward_manager",
                 "verl.experimental.reward_loop.reward_manager.base", "ifeval_oi", "ifeval_oi.verifier"):
        modules[name] = ModuleType(name)
    modules["verl"].DataProto = object
    modules["verl.experimental.reward_loop.reward_manager.base"].RewardManagerBase = FakeBase
    modules["ifeval_oi.verifier"].remove_thinking_section = lambda text, **kwargs: text
    modules["ifeval_oi.verifier"].score_ifeval = lambda *args, **kwargs: 0.5
    modules["aiohttp"].ClientTimeout = lambda **kwargs: kwargs
    spec = importlib.util.spec_from_file_location("intentcheck_reward_under_test", ROOT / "if_rlvr/if_llm_verifier_reward_manager.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    module._DEPS_OK = True
    return module


M = load_manager()


class Vec:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def __getitem__(self, key):
        return Vec(self.values[key]) if isinstance(key, slice) else self.values[key]

    def sum(self):
        return SimpleNamespace(item=lambda: sum(self.values))


class Batch:
    def __init__(self, phase="combined", eligible=False, has_x=True):
        fields = {"reward_model": {"ground_truth": {}},
                  "if_llm_verifier_phase": phase, "if_llm_verifier_eligible": eligible}
        if has_x:
            fields["ppl_prompt"] = [{"role": "user", "content": "Original x only"}]
        fields["raw_prompt"] = [{"role": "user", "content": "x plus constraint c"}]
        self.item = SimpleNamespace(batch={"responses": Vec([1, 2]), "attention_mask": Vec([1, 1, 1])},
                                    non_tensor_batch=fields)

    def __getitem__(self, key):
        return self if isinstance(key, slice) else self.item


class Reply:
    def __init__(self, content, status=200):
        self.content, self.status = content, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return json.dumps({"choices": [{"message": {"content": self.content}}]})


class Session:
    def __init__(self, content, calls, status=200):
        self.content, self.calls, self.status = content, calls, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, endpoint, json):
        self.calls.append((endpoint, json))
        return Reply(self.content, self.status)


class RewardTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, mode="intentcheck", **extra):
        kwargs = dict(if_llm_verifier_mode=mode, if_llm_verifier_model="Tulu",
                      if_llm_verifier_base_url="http://test/v1", if_llm_verifier_threshold=9,
                      if_llm_verifier_response_format=True, if_llm_verifier_enable_thinking="false",
                      if_llm_verifier_max_retries=0)
        kwargs.update(extra)
        config = AttrDict(reward=AttrDict(reward_kwargs=kwargs, reward_model=AttrDict(model_path="Tulu")), data={})
        return M.IFLLMVerifierRewardManager(config, SimpleNamespace(decode=lambda *a, **k: "Final answer"))

    async def test_reward_matrix_and_threshold_independence(self):
        for c in (0.0, 0.2, 0.5, 1.0):
            for score in (0, 1, None):
                with self.subTest(c=c, score=score):
                    manager = self.manager(if_llm_verifier_threshold=99)
                    manager._judge = AsyncMock(return_value=(score, "judgment", "parse error" if score is None else None))
                    with patch.object(M, "score_ifeval", return_value=c):
                        result = await manager.run_single(Batch())
                    self.assertAlmostEqual(result["reward_score"], c + (0.1 if c > 0 and score == 1 else 0))
                    self.assertEqual(manager._judge.await_count, int(c > 0))
                    info = result["reward_extra_info"]
                    self.assertEqual(info["llm_verifier_pass"], float(c > 0 and score == 1))
                    if c > 0:
                        manager._judge.assert_awaited_with("Original x only", "Final answer")

    async def test_legacy_threshold_and_errors_retain_constraint_reward(self):
        for score in (1, 8, 9, 10, None):
            manager = self.manager(mode="geval")
            manager._judge = AsyncMock(return_value=(score, None, "error" if score is None else None))
            with patch.object(M, "score_ifeval", return_value=0.5):
                result = await manager.run_single(Batch())
            self.assertAlmostEqual(result["reward_score"], 0.6 if score is not None and score >= 9 else 0.5)

    async def test_intentcheck_request_uses_paper_prompt_not_json(self):
        manager = self.manager()
        calls = []
        with patch.object(M.aiohttp, "ClientSession", create=True,
                          side_effect=lambda **k: Session("1. All checks pass.\n**Final Verification:** YES", calls)):
            score, raw, error = await manager._judge("x {input}", "response {answer}")
        self.assertEqual((score, error), (1, None))
        payload = calls[0][1]
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("<Instruction>\nx {input}\n</Instruction>", payload["messages"][0]["content"])
        self.assertIn("<Response>\nresponse {answer}\n</Response>", payload["messages"][0]["content"])
        self.assertNotIn("1 to 10", payload["messages"][0]["content"])

    async def test_network_no_parse_failure_and_http_error_are_distinguished(self):
        for content, status, expected in (("Final Verification: NO", 200, 0),
                                          ('1. Yes 2. Yes {"Score":10}', 200, None),
                                          (None, 200, None), ("Final Verification: YES", 500, None)):
            with self.subTest(content=content, status=status):
                manager = self.manager()
                calls = []
                with patch.object(M.aiohttp, "ClientSession", create=True,
                                  side_effect=lambda **k: Session(content, calls, status)):
                    score, raw, error = await manager._judge("x", "y")
                self.assertEqual(score, expected)
                self.assertEqual(error is not None, expected is None)

    async def test_legacy_json_request_and_parser_are_unchanged(self):
        manager = self.manager(mode="geval")
        calls = []
        with patch.object(M.aiohttp, "ClientSession", create=True,
                          side_effect=lambda **k: Session('{"Score": 9}', calls)):
            score, raw, error = await manager._judge("x", "y")
        self.assertEqual((score, error), (9, None))
        self.assertEqual(calls[0][1]["response_format"], {"type": "json_object"})
        self.assertEqual(calls[0][1]["messages"][0]["content"], M.DEFAULT_JUDGE_PROMPT.format(prompt="x", response="y"))

    async def test_default_mode_stays_geval_and_invalid_mode_fails(self):
        with patch.dict(M.os.environ, {}, clear=True):
            self.assertEqual(self.manager(if_llm_verifier_mode=None).mode, "geval")
            with self.assertRaises(ValueError):
                self.manager(mode="misspelled")

    async def test_missing_x_fails_instead_of_falling_back_to_x_plus_c(self):
        manager = self.manager()
        with self.assertRaises(KeyError):
            await manager.run_single(Batch(has_x=False))

    async def test_legacy_constraint_and_anchor_fallback_phases_unchanged(self):
        manager = self.manager(mode="geval", if_llm_verifier_anchor_fallback_only=True)
        manager._judge = AsyncMock(return_value=(10, "judgment", None))
        for phase, eligible, expected, called in (("combined", True, 0.5, False),
                                                 ("constraint", True, 0.5, False),
                                                 ("verifier", False, 0, False),
                                                 ("verifier", True, 0.1, True)):
            manager._judge.reset_mock()
            with patch.object(M, "score_ifeval", return_value=0.5):
                result = await manager.run_single(Batch(phase=phase, eligible=eligible))
            self.assertAlmostEqual(result["reward_score"], expected)
            self.assertEqual(manager._judge.await_count, int(called))


if __name__ == "__main__":
    unittest.main()
