# Copyright 2024 The Google Research Authors.
# Copyright 2024 AllenAI (open-instruct).
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
"""Instruction-following (IFEval) verifier, vendored from AllenAI open-instruct.

This reproduces ``open_instruct.ground_truth_utils.IFEvalVerifier`` exactly (the
``ifeval`` verifier used by ``scripts/train/rlvr/valpy_if_grpo_fast.sh`` on the
``allenai/IF_multi_constraints_upto5`` dataset), so that verl RLVR training scores
instruction-following outputs identically to open-instruct.

The ground-truth ``label`` is a string-encoded one-element list of dicts:

    "[{'instruction_id': ['detectable_format:title', ...], 'kwargs': [None, ...]}]"

``score_ifeval`` returns the fraction of constraints satisfied, in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import ast
import json
import logging

from . import instructions_registry

logger = logging.getLogger(__name__)


def _strip_gemma_thought_channel(prediction: str) -> str:
    # Gemma thinking uses: <|channel>thought\n...<channel|>[final answer].
    # When disabled, some variants may emit an empty thought block before the answer.
    for start_marker in ("<|channel>thought\n", "<|channel>thought"):
        start = prediction.find(start_marker)
        if start < 0:
            continue
        end = prediction.find("<channel|>", start + len(start_marker))
        if end >= 0:
            return prediction[end + len("<channel|>") :]
    return prediction


def _has_gemma_thought_channel(prediction: str) -> bool:
    return _strip_gemma_thought_channel(prediction) != prediction


def remove_thinking_section(prediction: str, require_think_end: bool = False) -> str | None:
    """Strip a reasoning/thinking section and answer tags before verification.

    Verbatim from open_instruct.ground_truth_utils.remove_thinking_section. For
    Qwen3 with ``enable_thinking=True`` the response is ``<think>...</think>...``;
    splitting on ``</think>`` and taking the last segment removes the reasoning
    tokens (and the ``</think>`` marker itself). For ``enable_thinking=False`` the
    response contains no ``</think>`` and is returned unchanged.
    """
    prediction = prediction.replace("<|assistant|>", "").strip()
    has_qwen_think_end = "</think>" in prediction
    has_gemma_thought = _has_gemma_thought_channel(prediction)
    if require_think_end and not (has_qwen_think_end or has_gemma_thought):
        return None
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    prediction = _strip_gemma_thought_channel(prediction)
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    prediction = prediction.replace("<|think|>", "")
    return prediction.strip()


def score_ifeval(prediction: str, label, require_think_end: bool = False) -> float:
    """Score one instruction-following response against its constraint set.

    Faithful reproduction of ``IFEvalVerifier.__call__`` (open-instruct). Returns a
    float in ``[0.0, 1.0]`` equal to the fraction of constraints the (thinking-
    stripped) prediction satisfies.

    Args:
        prediction: The decoded model output (special tokens already stripped, as
            in open-instruct ``batch_decode(..., skip_special_tokens=True)``).
        label: The ground-truth constraint spec. A string-encoded list of dicts
            (as stored in the dataset); a pre-parsed list/dict is also accepted.
    """
    instruction_dict = instructions_registry.INSTRUCTION_DICT
    # Parse the ground truth. open-instruct stores it as a string and does
    # ``ast.literal_eval(label)[0]``; we accept an already-parsed list/dict too,
    # which is behaviourally identical for the string inputs used in training.
    if isinstance(label, str):
        constraint_dict = ast.literal_eval(label)
    else:
        constraint_dict = label
    constraint_dict = constraint_dict[0]
    if isinstance(constraint_dict, str):
        constraint_dict = json.loads(constraint_dict)
    answer = remove_thinking_section(prediction, require_think_end=require_think_end)
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]
    rewards = []
    if answer is None:
        logger.warning("Missing </think> in reasoning response received for IFEvalVerifier.")
        return 0.0
    if len(prediction) == 0 or len(answer) == 0:
        logger.warning("Empty prediction received for IFEvalVerifier.")
        return 0.0
    for instruction_key, args in zip(instruction_keys, args_list):
        if args is None:
            args = {}
        args = {k: v for k, v in args.items() if v is not None}
        instruction_cls = instruction_dict[instruction_key]
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)
        if prediction.strip() and instruction_instance.check_following(answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return sum(rewards) / max(len(rewards), 1)
