# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import math
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @staticmethod
    def _find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
        # ##6/3 ppl## Locate the </think> marker in token space so y is final answer only.
        if not pattern or len(pattern) > len(sequence):
            return None
        for idx in range(len(sequence) - len(pattern) + 1):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return None

    def _extract_final_answer_ids(self, response_ids: list[int], prompt_ids: list[int]) -> tuple[list[int], bool]:
        # ##6/3 ppl## If </think> is generated, y is the tokens after it.
        # ##6/3 ppl## If enable_thinking=False already closed </think> in the prompt, y is the whole response.
        think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        marker_pos = self._find_subsequence(response_ids, think_end_ids)
        if marker_pos is not None:
            final_answer_ids = response_ids[marker_pos + len(think_end_ids) :]
            return final_answer_ids, len(final_answer_ids) > 0
        recent_prompt_ids = prompt_ids[-max(64, len(think_end_ids) + 8) :]
        if self._find_subsequence(recent_prompt_ids, think_end_ids) is not None:
            return response_ids, len(response_ids) > 0
        return [], False

    @staticmethod
    def _empty_ppl_fields(prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_ppl": float("inf"),
            f"{prefix}_mean_logprob": None,
            f"{prefix}_nll": None,
            f"{prefix}_token_count": 0,
            f"{prefix}_eligible": False,
        }

    @staticmethod
    def _constraint_free_prompt_text(ppl_prompt) -> str:
        if not ppl_prompt:
            return ""
        user_parts = [str(message.get("content", "")) for message in ppl_prompt if message.get("role") == "user"]
        if user_parts:
            return "\n".join(part for part in user_parts if part).strip()
        return "\n".join(str(message.get("content", "")) for message in ppl_prompt).strip()

    async def _score_continuation_ppl(
        self, prefix_ids: list[int], continuation_ids: list[int], metric_prefix: str
    ) -> dict[str, Any]:
        if not prefix_ids or not continuation_ids:
            return self._empty_ppl_fields(metric_prefix)

        sequence_ids = prefix_ids + continuation_ids
        score_output: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": 0},
        )
        prompt_logprobs = score_output.extra_fields.get("prompt_logprobs") or []
        start = max(len(prefix_ids) - 1, 0)
        rows = prompt_logprobs[start : start + len(continuation_ids)]
        token_logprobs = [float(row[0]) for row in rows if row and row[0] is not None]
        if len(token_logprobs) != len(continuation_ids):
            raise ValueError(
                f"{metric_prefix} PPL logprob length mismatch: "
                f"got {len(token_logprobs)}, expected {len(continuation_ids)}"
            )
        mean_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)
        nll = -sum(token_logprobs)
        return {
            f"{metric_prefix}_ppl": float(math.exp(-mean_logprob)),
            f"{metric_prefix}_mean_logprob": float(mean_logprob),
            f"{metric_prefix}_nll": float(nll),
            f"{metric_prefix}_token_count": len(token_logprobs),
            f"{metric_prefix}_eligible": True,
        }

    async def _compute_bidirectional_ppl(
        self, ppl_prompt, response_ids: list[int], prompt_ids: list[int]
    ) -> dict[str, Any]:
        # ##6/3 ppl## Score p(y|x) and p(x|y), where y is the final answer and x is the constraint-free prompt.
        final_answer_ids, eligible = self._extract_final_answer_ids(response_ids, prompt_ids)
        defaults = {
            **self._empty_ppl_fields("p_y_given_x"),
            **self._empty_ppl_fields("p_x_given_y"),
        }
        if not eligible or not ppl_prompt:
            return defaults

        extra_fields = dict(defaults)
        try:
            x_prompt_ids = await self.apply_chat_template(
                list(ppl_prompt),
                apply_chat_template_kwargs={"enable_thinking": False},
            )
            extra_fields.update(await self._score_continuation_ppl(x_prompt_ids, final_answer_ids, "p_y_given_x"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("##6/3 ppl## p(y|x) PPL scoring failed: %s", exc)

        try:
            final_answer_text = self.tokenizer.decode(final_answer_ids, skip_special_tokens=True).strip()
            x_text = self._constraint_free_prompt_text(ppl_prompt)
            if not final_answer_text or not x_text:
                return extra_fields
            y_prompt_ids = await self.apply_chat_template(
                [{"role": "user", "content": final_answer_text}],
                apply_chat_template_kwargs={"enable_thinking": False},
            )
            x_response_ids = self.tokenizer.encode(x_text, add_special_tokens=False)
            extra_fields.update(await self._score_continuation_ppl(y_prompt_ids, x_response_ids, "p_x_given_y"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("##6/3 ppl## p(x|y) PPL scoring failed: %s", exc)

        return extra_fields

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        response_ids = token_output.token_ids[: self.response_length]
        response_mask = [1] * len(response_ids)

        # ##6/3 ppl## Score all rollouts while vLLM is still awake; trainer later gates by old reward.
        ppl_extra_fields = await self._compute_bidirectional_ppl(kwargs.get("ppl_prompt"), response_ids, prompt_ids)
        extra_fields = dict(token_output.extra_fields)
        extra_fields.update(ppl_extra_fields)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=token_output.log_probs[: self.response_length] if token_output.log_probs else None,
            routed_experts=(
                token_output.routed_experts[: len(prompt_ids) + self.response_length]
                if token_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
