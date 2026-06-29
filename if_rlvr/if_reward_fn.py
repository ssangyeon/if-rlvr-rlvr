# Copyright 2024 AllenAI (open-instruct); 2025 verl IF-RLVR migration.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Orthogonal verl reward function for instruction-following (IFEval).

This is the **recommended** integration: it plugs into verl through the stable, public
``custom_reward_function`` hook, so it works with ANY verl reward manager (naive / dapo /
prime / batch / experimental), ANY RL algorithm (GRPO / PPO / RLOO / DAPO / ...), and across
verl versions — the reward stays orthogonal to verl's training internals and patches.

It has NO ``verl`` imports: it only needs the sibling ``ifeval_oi`` verifier package, which it
locates relative to this file. The whole ``if_rlvr/`` directory is self-contained and can
be placed anywhere (inside or outside the verl tree) and referenced by absolute path.

Wire it in (Hydra), keeping verl's standard reward manager (naive) and any algorithm:
    custom_reward_function.path=<abs path to this file>
    custom_reward_function.name=compute_score

Reward = fraction of IFEval constraints satisfied (in [0,1]); the thinking section is stripped
before verification (Qwen3 ``enable_thinking``). This reproduces open-instruct's ``ifeval``
verification score exactly. (The non-stop / truncation penalty from valpy_if_grpo_fast.sh needs
response token ids, which a ``custom_reward_function`` does not receive; for that, use the optional
``if_reward_manager.py`` instead — it couples to verl's reward-manager API. See README.)
"""

from __future__ import annotations

import os
import sys

# Make the self-contained verifier package importable regardless of where this file is loaded
# from (verl loads custom_reward_function by file path, without package context).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ifeval_oi.verifier import score_ifeval  # noqa: E402  (after sys.path bootstrap)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, verification_reward=1.0, **kwargs):
    """verl ``custom_reward_function`` entry point.

    Args:
        data_source: dataset tag (``"ifeval"``); unused for dispatch (this fn is IF-specific).
        solution_str: decoded model response (thinking section stripped inside ``score_ifeval``).
        ground_truth: the string-encoded IFEval constraint spec
            ``"[{'instruction_id': [...], 'kwargs': [...]}]"``.
        extra_info: unused.
        verification_reward: optional scalar multiplier (default 1.0 -> raw fraction in [0,1],
            which is verl-idiomatic and, under GRPO std-normalization, equivalent to any positive
            scale; open-instruct used 10.0). Override via
            ``custom_reward_function.reward_kwargs.verification_reward=...``.

    Returns:
        dict with mandatory ``"score"`` key (+ ``"acc"`` = the raw fraction, for logging).
    """
    require_think_end = _env_bool("IF_REQUIRE_THINK_END_FOR_REWARD", False)
    score = score_ifeval(solution_str, ground_truth, require_think_end=require_think_end)
    return {"score": float(verification_reward) * score, "acc": score}
