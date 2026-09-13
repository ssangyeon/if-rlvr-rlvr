# Copyright 2025 Individual Contributor: Justin Seo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise reference likelihood rewards on CPU without starting Ray or loading models."""

import ast
import asyncio
import json
import logging
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_trainer_functions():
    """Compile the production helpers without importing the distributed trainer dependencies."""
    path = ROOT / "verl/trainer/ppo/ray_trainer.py"
    module = ast.parse(path.read_text(), filename=str(path))
    names = {
        "normalize_if_ppl_reward_strategy",
        "normalize_bool_config",
        "normalize_clipped_rollout_mode",
        "get_clipped_rollout_mask",
        "apply_clipped_rollout_reward_mode",
        "_apply_if_likelihood_reward",
        "apply_if_inverse_ppl_reward",
        "apply_if_raw_likelihood_reward",
        "_coerce_token_ids",
        "_build_ref_policy_scoring_batch",
        "_score_continuations_with_ref_policy",
        "_compute_anchor_ppl_with_ref_policy",
    }
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name in names]
    constants = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"_CLIPPED_ROLLOUT_MODES", "_IF_PPL_REWARD_STRATEGIES"}
            for target in node.targets
        )
    ]
    isolated = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *constants,
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {
        "math": math,
        "np": np,
        "torch": torch,
        "DataProto": SimpleNamespace(from_dict=lambda tensors: SimpleNamespace(batch=tensors)),
    }
    position_path = ROOT / "verl/utils/model.py"
    position_module = ast.parse(position_path.read_text(), filename=str(position_path))
    position_helper = next(
        node
        for node in position_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_position_id_with_mask"
    )
    exec(compile(ast.Module(body=[position_helper], type_ignores=[]), str(position_path), "exec"), namespace)
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace


def load_agent_loop_functions():
    """Load the Qwen answer extractor and real rollout path with CPU transport stubs."""
    path = ROOT / "verl/experimental/agent_loop/single_turn_agent_loop.py"
    module = ast.parse(path.read_text(), filename=str(path))
    names = {"_strip_leading_whitespace_ids", "_extract_final_answer_ids", "run"}
    functions = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    isolated = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *functions,
        ],
        type_ignores=[],
    )
    namespace = {
        "rollout_trace_op": lambda function: function,
        "simple_timer": lambda *args: nullcontext(),
        "uuid4": lambda: SimpleNamespace(hex="cpu-test"),
        "AgentLoopOutput": SimpleNamespace,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace


HELPERS = load_trainer_functions()
AGENT_HELPERS = load_agent_loop_functions()
APPLY = HELPERS["apply_if_inverse_ppl_reward"]
APPLY_RAW = HELPERS["apply_if_raw_likelihood_reward"]


class CharacterTokenizer:
    """Exercise decoded think-marker splitting without downloading a tokenizer."""

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def generate_test_rollout(strategy, response_text, prompt_text="x+c<think>"):
    """Run production answer extraction/preparation with only generation mocked."""
    tokenizer = CharacterTokenizer()
    token_output = SimpleNamespace(
        token_ids=tokenizer.encode(response_text),
        num_preempted=0,
        extra_fields={},
        log_probs=None,
        routed_experts=None,
    )
    loop = SimpleNamespace(
        config={"if_ppl_reward_strategy": strategy},
        tokenizer=tokenizer,
        response_length=128,
        allow_missing_think_final_answer=False,
        anchor_precompute_final_answer_only=False,
        defer_ppl_to_ref_policy=True,
        process_multi_modal_info=AsyncMock(return_value={}),
        _get_mm_processor_kwargs=lambda audios: {},
        apply_chat_template=AsyncMock(return_value=tokenizer.encode(prompt_text)),
        _extract_gemma_final_answer_ids=lambda response_ids: None,
        _build_ppl_x_prompt_ids=AsyncMock(return_value=[11, 12]),
        server_manager=SimpleNamespace(generate=AsyncMock(return_value=token_output)),
    )
    for name in ("_strip_leading_whitespace_ids", "_extract_final_answer_ids"):
        setattr(loop, name, MethodType(AGENT_HELPERS[name], loop))
    x_prompt = [{"role": "user", "content": "x"}]
    output = asyncio.run(
        AGENT_HELPERS["run"](loop, {}, raw_prompt=[{"role": "user", "content": "x+c"}], ppl_prompt=x_prompt)
    )
    loop._build_ppl_x_prompt_ids.assert_awaited_once_with(x_prompt)
    return output


@pytest.fixture(params=["inverse_ppl", "raw_likelihood"])
def likelihood_reward(request):
    field = "likelihood" if request.param == "inverse_ppl" else "raw_likelihood"
    return HELPERS[f"apply_if_{request.param}_reward"], f"if_p_y_given_x_{field}"


def make_batch(nlls, counts, constraints=None, lengths=None, width=4):
    """Build only the batch fields the production reward helper consumes."""
    size = len(nlls)
    lengths = [2] * size if lengths is None else lengths
    constraints = [0.0] * size if constraints is None else constraints
    response_mask = torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    reward = torch.zeros(size, width)
    for row, (length, constraint) in enumerate(zip(lengths, constraints, strict=True)):
        if length:
            reward[row, length - 1] = constraint
    batch = SimpleNamespace(
        batch={"response_mask": response_mask, "responses": torch.zeros(size, width, dtype=torch.long)},
        non_tensor_batch={"p_y_given_x_nll": nlls, "p_y_given_x_token_count": counts},
    )
    return batch, reward


def test_direct_addition_includes_constraint_failures_and_ignores_anchors():
    batch, reward = make_batch([2 * math.log(2)] * 3, [2] * 3, constraints=[0, 0.5, 1], lengths=[1, 2, 3])
    batch.non_tensor_batch.update({"p_y_ref_given_x_nll": [np.nan] * 3, "ppl_y_ref0_ids": [None] * 3})
    original = reward.clone()
    updated, metrics = APPLY(batch, reward)
    torch.testing.assert_close(updated.sum(-1), torch.tensor([0.5, 1.0, 1.5]))
    delta = torch.zeros_like(reward)
    delta[torch.arange(3), torch.tensor([0, 1, 2])] = 0.5
    torch.testing.assert_close(updated - reward, delta)
    torch.testing.assert_close(reward, original)
    assert metrics["if_p_y_given_x_likelihood/likelihood_reward/mean"] == pytest.approx(0.5)


def test_length_normalized_likelihood_survives_large_summed_nll():
    batch, reward = make_batch([1.0, 1000.0, 2048.0], [1, 1000, 2048])
    updated, _ = APPLY(batch, reward)
    torch.testing.assert_close(updated.sum(-1), torch.full((3,), math.exp(-1)))


def test_continuous_score_increases_with_predictability():
    batch, reward = make_batch([0, 1, 2, 10000], [1] * 4)
    updated, _ = APPLY(batch, reward)
    torch.testing.assert_close(updated.sum(-1), torch.tensor([1, math.exp(-1), math.exp(-2), 0]))


def test_raw_likelihood_adds_token_probability_product_even_when_constraints_fail():
    probability = 0.5 * 0.25 * 0.1
    batch, reward = make_batch([-math.log(probability)] * 3, [3] * 3, constraints=[0, 0.5, 1], lengths=[3] * 3)
    batch.non_tensor_batch.update({"p_y_ref_given_x_nll": [np.nan] * 3, "ppl_y_ref0_ids": [None] * 3})
    original = reward.clone()
    updated, metrics = APPLY_RAW(batch, reward)
    torch.testing.assert_close(updated.sum(-1), torch.tensor([probability, 0.5 + probability, 1 + probability]))
    delta = torch.zeros_like(reward)
    delta[:, 2] = probability
    torch.testing.assert_close(updated - reward, delta)
    torch.testing.assert_close(reward, original)
    assert metrics["if_p_y_given_x_raw_likelihood/likelihood/mean"] == pytest.approx(probability)


def test_raw_likelihood_uses_summed_nll_without_length_normalization():
    batch, reward = make_batch([math.log(4)] * 3, [1, 100, 2048])
    updated, _ = APPLY_RAW(batch, reward)
    torch.testing.assert_close(updated.sum(-1), torch.full((3,), 0.25))


def test_raw_likelihood_diagnoses_underflow_and_rounding_without_changing_the_reward():
    # exp(-1000) underflows in float64; exp(-230) is lost on the float32 cast;
    # exp(-20) survives that cast but is rounded away when added to C=1.
    batch, reward = make_batch([1000, 230, 20], [1000, 230, 20], constraints=[1] * 3)
    updated, metrics = APPLY_RAW(batch, reward)
    torch.testing.assert_close(updated, reward)
    assert batch.non_tensor_batch["p_y_given_x_raw_likelihood"].tolist() == pytest.approx(
        [0, math.exp(-230), math.exp(-20)], abs=0
    )
    prefix = "if_p_y_given_x_raw_likelihood"
    assert metrics[f"{prefix}/zero_likelihood_count"] == 1
    assert metrics[f"{prefix}/term_underflow_count"] == 1
    assert metrics[f"{prefix}/rounded_away_count"] == 1


@pytest.mark.parametrize("coeff", [0, 0.1, 1])
def test_coefficient_scales_only_the_likelihood_term(coeff, likelihood_reward):
    apply_reward, _ = likelihood_reward
    batch, reward = make_batch([math.log(4)], [1], constraints=[0.5])
    updated, _ = apply_reward(batch, reward, ppl_reward_coeff=coeff)
    assert updated.sum().item() == pytest.approx(0.5 + coeff / 4)


@pytest.mark.parametrize("coeff", [-1, np.nan, np.inf])
def test_invalid_coefficient_is_rejected(coeff, likelihood_reward):
    apply_reward, _ = likelihood_reward
    batch, reward = make_batch([1], [1])
    with pytest.raises(ValueError, match="finite, nonnegative"):
        apply_reward(batch, reward, ppl_reward_coeff=coeff)


def test_invalid_or_empty_scored_answers_do_not_receive_reward(likelihood_reward):
    apply_reward, prefix = likelihood_reward
    nlls = [np.inf, np.nan, None, "bad", -1, 0, 0, 0]
    counts = [1, 1, 1, 1, 1, 0, np.inf, 1]
    batch, reward = make_batch(nlls, counts, constraints=[0.5] * 8, lengths=[2] * 7 + [0])
    updated, metrics = apply_reward(batch, reward)
    torch.testing.assert_close(updated, reward)
    assert metrics[f"{prefix}/invalid_count"] == 8
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("mode, expected", [("use", [1.5, 0.5]), ("zero", [0, 0.5])])
def test_clipped_zero_mode_cannot_be_undone_by_likelihood(mode, expected, likelihood_reward):
    apply_reward, prefix = likelihood_reward
    batch, reward = make_batch([math.log(2)] * 2, [1] * 2, constraints=[1, 0], lengths=[4, 2])
    reward, _ = HELPERS["apply_clipped_rollout_reward_mode"](batch, reward, mode)
    updated, metrics = apply_reward(batch, reward, clipped_rollout_mode=mode)
    torch.testing.assert_close(updated.sum(-1), torch.tensor(expected, dtype=torch.float32))
    assert metrics[f"{prefix}/clipped_excluded_count"] == int(mode == "zero")


def test_missing_or_misaligned_scores_fail_instead_of_silently_using_constraints_only(likelihood_reward):
    apply_reward, _ = likelihood_reward
    batch, reward = make_batch([1], [1])
    batch.non_tensor_batch.pop("p_y_given_x_nll")
    with pytest.raises(ValueError, match="missing fields"):
        apply_reward(batch, reward)
    batch.non_tensor_batch["p_y_given_x_nll"] = [1, 2]
    with pytest.raises(ValueError, match="batch size"):
        apply_reward(batch, reward)


@pytest.mark.parametrize("with_anchor_fields", [False, True])
def test_reference_scoring_uses_only_x_and_rollout_answer(with_anchor_fields):
    calls = []

    def score(prefixes, continuations):
        calls.append((prefixes, continuations))
        return np.array([math.log(2)]), np.array([1]), np.array([2.0])

    scorer = SimpleNamespace(
        config={"if_ref_policy_anchor_ppl": True},
        _validate_ppl_x_prefix_ids=lambda prefixes: None,
        _score_continuations_with_ref_policy=score,
    )
    batch = SimpleNamespace(non_tensor_batch={"ppl_x_prompt_ids": [[11, 12]], "ppl_y_final_answer_ids": [[13]]})
    if with_anchor_fields:
        batch.non_tensor_batch.update({"ppl_y_ref0_ids": [[99]], "ppl_y_ref1_ids": [[98]]})
    metrics = HELPERS["_compute_anchor_ppl_with_ref_policy"](scorer, batch, include_anchors=False)
    assert calls == [([[11, 12]], [[13]])]
    assert batch.non_tensor_batch["p_y_given_x_nll"][0] == pytest.approx(math.log(2))
    assert "p_y_ref_given_x_nll" not in batch.non_tensor_batch
    assert not any("missing_cache" in key for key in metrics)


@pytest.mark.parametrize("strategy", ["inverse_ppl", "raw_likelihood"])
def test_both_baselines_score_only_the_extracted_answer_conditioned_on_x(strategy):
    calls = []
    output = generate_test_rollout(strategy, "secret reasoning</think>\n\nYES")
    answer_ids = CharacterTokenizer().encode("YES")
    assert output.extra_fields["ppl_y_final_answer_ids"] == answer_ids
    width = len(output.response_ids) + 2
    batch, reward = make_batch([None], [None], constraints=[0.5], lengths=[len(output.response_ids)], width=width)
    batch.batch["responses"] = torch.tensor([output.response_ids + [0, 0]])
    batch.batch["prompts"] = torch.tensor([output.prompt_ids])
    for key in ("ppl_x_prompt_ids", "ppl_y_final_answer_ids"):
        batch.non_tensor_batch[key] = [output.extra_fields[key]]
    batch.non_tensor_batch.update({"ppl_y_ref0_ids": [[88]], "ppl_y_ref1_ids": [[89]]})
    builder = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=4, response_length=width))
        ),
        tokenizer=SimpleNamespace(pad_token_id=0),
        ref_in_actor=False,
        ref_policy_wg=object(),
        _get_dp_size=lambda worker, role: 2,
    )
    builder._coerce_token_ids = MethodType(HELPERS["_coerce_token_ids"], builder)

    def compute_ref_log_prob(scoring_batch):
        calls.append(scoring_batch)
        # An extra row pads the scoring batch for DP; it must not add another reward.
        assert scoring_batch.batch["prompts"].tolist() == [[0, 0, 11, 12]] * 2
        assert scoring_batch.batch["responses"].tolist() == [answer_ids + [0] * (width - 3)] * 2
        assert scoring_batch.batch["response_mask"].tolist() == [[1, 1, 1] + [0] * (width - 3)] * 2
        assert scoring_batch.batch["input_ids"].tolist() == [[0, 0, 11, 12] + answer_ids + [0] * (width - 3)] * 2
        torch.testing.assert_close(scoring_batch.batch["temperature"], torch.ones(2))
        log_probs = [math.log(0.5), math.log(0.25), math.log(0.1)] + [-999] * (width - 3)
        return SimpleNamespace(batch={"ref_log_prob": torch.tensor([log_probs] * 2)})

    scorer = SimpleNamespace(
        config={"if_ref_policy_anchor_ppl": True},
        _validate_ppl_x_prefix_ids=lambda prefixes: None,
        _build_ref_policy_scoring_batch=MethodType(HELPERS["_build_ref_policy_scoring_batch"], builder),
        _compute_ref_log_prob=compute_ref_log_prob,
    )
    scorer._score_continuations_with_ref_policy = MethodType(HELPERS["_score_continuations_with_ref_policy"], scorer)
    HELPERS["_compute_anchor_ppl_with_ref_policy"](scorer, batch, include_anchors=False)
    assert len(calls) == 1
    assert batch.non_tensor_batch["p_y_given_x_nll"].tolist() == pytest.approx([-math.log(0.5 * 0.25 * 0.1)])
    assert batch.non_tensor_batch["p_y_given_x_token_count"].tolist() == [3]
    assert "p_y_ref_given_x_nll" not in batch.non_tensor_batch
    updated, _ = HELPERS[f"apply_if_{strategy}_reward"](batch, reward)
    probability = 0.5 * 0.25 * 0.1
    expected = probability if strategy == "raw_likelihood" else probability ** (1 / 3)
    assert updated.sum().item() == pytest.approx(0.5 + expected)


@pytest.mark.parametrize("strategy", ["inverse_ppl", "raw_likelihood"])
@pytest.mark.parametrize("response", ["unfinished reasoning", "reasoning</think>\n\n"])
def test_missing_final_answer_never_falls_back_to_scoring_reasoning(strategy, response):
    output = generate_test_rollout(strategy, response)
    assert output.extra_fields["ppl_y_final_answer_ids"] == []
    batch, reward = make_batch([None], [None], constraints=[0.5])
    for key in ("ppl_x_prompt_ids", "ppl_y_final_answer_ids"):
        batch.non_tensor_batch[key] = [output.extra_fields[key]]
    scorer = SimpleNamespace(
        config={"if_ref_policy_anchor_ppl": True},
        _validate_ppl_x_prefix_ids=lambda prefixes: None,
        _build_ref_policy_scoring_batch=lambda prefixes, continuations: (None, None),
    )
    scorer._score_continuations_with_ref_policy = MethodType(HELPERS["_score_continuations_with_ref_policy"], scorer)
    HELPERS["_compute_anchor_ppl_with_ref_policy"](scorer, batch, include_anchors=False)
    updated, _ = HELPERS[f"apply_if_{strategy}_reward"](batch, reward)
    torch.testing.assert_close(updated, reward)


@pytest.mark.parametrize("strategy", ["inverse_ppl", "raw_likelihood"])
def test_nonreasoning_prefix_uses_the_response_as_the_final_answer(strategy):
    output = generate_test_rollout(strategy, "\n\nYES", prompt_text="x+c<think>\n</think>\n\n")
    assert output.extra_fields["ppl_y_final_answer_ids"] == CharacterTokenizer().encode("YES")


def test_existing_anchor_fallback_behavior_is_preserved():
    output = generate_test_rollout("anchor", "unfinished reasoning")
    assert output.extra_fields["ppl_y_final_answer_ids"] == CharacterTokenizer().encode("unfinished reasoning")


def test_anchor_scoring_remains_the_default():
    calls = []

    def score(prefixes, continuations):
        calls.append(continuations)
        return np.array([1.0]), np.array([1]), np.array([math.e])

    scorer = SimpleNamespace(
        config={"if_ref_policy_anchor_ppl": True},
        _validate_ppl_x_prefix_ids=lambda prefixes: None,
        _score_continuations_with_ref_policy=score,
    )
    batch = SimpleNamespace(
        non_tensor_batch={
            "ppl_x_prompt_ids": [[11]],
            "ppl_y_final_answer_ids": [[12]],
            "ppl_y_ref0_ids": [[13]],
            "ppl_y_ref1_ids": [[14]],
        }
    )
    HELPERS["_compute_anchor_ppl_with_ref_policy"](scorer, batch)
    assert calls == [[[12]], [[13]], [[14]]]


def test_unknown_reward_strategy_is_rejected():
    normalize = HELPERS["normalize_if_ppl_reward_strategy"]
    assert normalize("inverse_ppl") == "inverse_ppl"
    assert normalize("raw_likelihood") == "raw_likelihood"
    with pytest.raises(ValueError, match="if_ppl_reward_strategy"):
        normalize("likelihood_typo")


@pytest.mark.parametrize("strategy", ["inverse_ppl", "raw_likelihood"])
def test_launcher_disables_anchors_and_forwards_hydra_arguments(tmp_path, strategy):
    # Intercept the shared launcher; never initialize Ray, download data, or use GPUs.
    keys = [
        "IF_PPL_REWARD_STRATEGY",
        "PY_GIVEN_X_REWARD_COEFF",
        "PX_GIVEN_Y_REWARD_COEFF",
        "IF_REF_POLICY_ANCHOR_PPL",
        "IF_REF_ANCHOR_PRECOMPUTE",
        "IF_REF_ANCHOR_TRAIN_CACHED_ONLY",
        "IF_REF_PPL_GATE",
        "IF_REF_PPL_BASELINE",
        "IF_REF_PPL_ANCHOR",
        "EXPERIMENT_NAME",
    ]
    stub = tmp_path / "bash"
    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"print(json.dumps({{'env': {{key: os.environ[key] for key in {keys!r}}}, 'args': sys.argv[1:]}}))\n"
    )
    stub.chmod(0o755)
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("IF_", "PY_GIVEN_X", "EXPERIMENT_NAME"))
    }
    env.update({"PATH": f"{tmp_path}:{env['PATH']}", "VERL_DIR": str(ROOT), "IF_REF_ANCHOR_PRECOMPUTE": "true"})
    launcher = ROOT / f"if_rlvr/exps/bidirectional/qwen3_17b_{strategy}_nonreason.sh"
    result = subprocess.run(
        ["/bin/bash", str(launcher), "trainer.total_training_steps=1"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed["env"]["PY_GIVEN_X_REWARD_COEFF"] == "1.0"
    assert observed["env"]["IF_PPL_REWARD_STRATEGY"] == strategy
    assert observed["env"]["IF_REF_POLICY_ANCHOR_PPL"] == "true"
    for key in ("IF_REF_ANCHOR_PRECOMPUTE", "IF_REF_ANCHOR_TRAIN_CACHED_ONLY", "IF_REF_PPL_GATE"):
        assert observed["env"][key] == "false"
    for key in ("IF_REF_PPL_BASELINE", "IF_REF_PPL_ANCHOR"):
        assert observed["env"][key] == "0"
    assert observed["env"]["PX_GIVEN_Y_REWARD_COEFF"] == "0.0"
    assert "coeff1.0" in observed["env"]["EXPERIMENT_NAME"]
    assert strategy in observed["env"]["EXPERIMENT_NAME"]
    assert observed["args"][-1] == "trainer.total_training_steps=1"
