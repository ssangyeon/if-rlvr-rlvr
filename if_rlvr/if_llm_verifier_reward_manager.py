"""IF reward manager with an external LLM verifier bonus.

Reward = IFEval constraint score + bonus, where the bonus is applied only when
the constraint score is positive and the LLM judge score is at least the
configured threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from typing import Any

import aiohttp

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ifeval_oi.verifier import remove_thinking_section, score_ifeval  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


DEFAULT_JUDGE_PROMPT = """Evaluate the response provided below to determine if it meets the specified constraints related to
the following prompt. Provide an integer score from 1 to 10, taking into account its helpfulness,
relevance, accuracy, depth, creativity, and how well it conforms to the constraints. Here are the criteria
that you should score: 1. Helpfulness: Does the response address the user's needs and questions
effectively? 2. Relevance: Is the response directly related to the context of the dialog? 3. Accuracy:
Are the facts and information presented in the response correct? 4. Depth: Does the response cover
the topic thoroughly, with sufficient detail? 5. Creativity: Is the response original and engaging?
Prompt to Evaluate Against:
{prompt}

Response to Evaluate:
{response}

The evaluation must be structured in the following JSON format:
{{"Score": <An integer score from 1 to 10>}}"""


_DEPS_OK = False


def noop_compute_score(*args, **kwargs):  # noqa: ARG001
    """Hydra-loadable placeholder; IFLLMVerifierRewardManager does the scoring."""
    return {"score": 0.0}


def _verify_runtime_deps() -> None:
    global _DEPS_OK
    if _DEPS_OK:
        return
    try:
        import langdetect  # noqa: F401

        from ifeval_oi import instructions_util

        instructions_util.nltk.word_tokenize("This is a sentence. Here is another one.")
        instructions_util.count_words("two words")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "IFLLMVerifierRewardManager runtime dependency check failed on this node: "
            f"{type(exc).__name__}: {exc}. Install `langdetect immutabledict nltk` and the nltk "
            "'punkt'/'punkt_tab' data on EVERY ray worker node (or set NLTK_DATA to shared storage)."
        ) from exc
    _DEPS_OK = True


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_reward_kwargs(config) -> dict[str, Any]:
    try:
        return dict(config.reward.get("reward_kwargs", None) or {})
    except Exception:  # noqa: BLE001
        return {}


def _get_with_env(kwargs: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    value = kwargs.get(key, None)
    if value is not None:
        return value
    return os.getenv(env_name, default)


def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _prompt_to_text(raw_prompt: Any) -> str:
    if hasattr(raw_prompt, "tolist"):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, dict):
        raw_prompt = [raw_prompt]
    if isinstance(raw_prompt, (list, tuple)):
        messages = list(raw_prompt)
        if len(messages) == 1 and isinstance(messages[0], dict):
            return str(messages[0].get("content", ""))
        parts = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "message")
                content = message.get("content", "")
                parts.append(f"{role}:\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    try:
        return json.dumps(raw_prompt, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(raw_prompt)


def _strip_json_fence(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    return stripped


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {None, "text", "output_text"}:
                    parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _strip_reasoning_from_final_content(content: str) -> str:
    text = _content_to_text(content).strip()
    if not text:
        return ""

    final_channel = "<|channel|>final"
    if final_channel in text:
        text = text.rsplit(final_channel, 1)[-1]
        if "<|message|>" in text:
            text = text.split("<|message|>", 1)[-1]
        for stop_token in ("<|end|>", "<|return|>"):
            if stop_token in text:
                text = text.split(stop_token, 1)[0]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if text.lower().startswith("<think>"):
        return ""
    return text


def _message_final_content(message: dict[str, Any]) -> str:
    # For gpt-oss over Chat Completions, vLLM should expose the final answer in
    # message.content and reasoning in separate fields. Never fall back to
    # reasoning/reasoning_content for score parsing.
    return _strip_reasoning_from_final_content(_content_to_text(message.get("content")))


def extract_judge_score(raw_judgment: str) -> int:
    text = _strip_json_fence(raw_judgment)
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        score_value = parsed.get("Score", parsed.get("score"))
        if isinstance(score_value, bool):
            raise ValueError(f"Score must be an integer from 1 to 10: {raw_judgment[:200]}")
        if isinstance(score_value, str):
            score_text = score_value.strip()
            if score_text.isdigit():
                score_value = int(score_text)
            else:
                score_match = re.search(r"\b(10|[1-9])(?:\.0+)?(?:\s*/\s*10)?\b", score_text)
                if score_match:
                    score_value = int(score_match.group(1))
        if isinstance(score_value, float) and score_value.is_integer():
            score_value = int(score_value)
        if isinstance(score_value, int) and 1 <= score_value <= 10:
            return score_value
        raise ValueError(f"Score must be an integer from 1 to 10: {raw_judgment[:200]}")
    if isinstance(parsed, int) and not isinstance(parsed, bool) and 1 <= parsed <= 10:
        return parsed
    if isinstance(parsed, float) and parsed.is_integer() and 1 <= int(parsed) <= 10:
        return int(parsed)

    score_match = re.search(
        r'"?Score"?\s*[:=]\s*(10|[1-9])(?:\.0+)?(?:\s*/\s*10)?\b',
        text,
        flags=re.IGNORECASE,
    )
    if not score_match:
        score_match = re.search(r"\b(10|[1-9])\b", text)
    if score_match:
        return int(score_match.group(1))
    raise ValueError(f"Judge did not return a parseable score: {raw_judgment[:200]}")


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class IFLLMVerifierRewardManager(RewardManagerBase):
    """Instruction-following reward with a G-Eval style LLM verifier bonus."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        _verify_runtime_deps()
        kwargs = _get_reward_kwargs(config)

        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.verification_reward = float(kwargs.get("verification_reward", 1.0))
        self.bonus = float(_get_with_env(kwargs, "if_llm_verifier_bonus", "IF_LLM_VERIFIER_BONUS", 0.1))
        self.threshold = int(_get_with_env(kwargs, "if_llm_verifier_threshold", "IF_LLM_VERIFIER_THRESHOLD", 5))
        self.model = str(
            _get_with_env(
                kwargs,
                "if_llm_verifier_model",
                "IF_LLM_VERIFIER_MODEL",
                getattr(config.reward.reward_model, "model_path", None) or "openai/gpt-oss-120b",
            )
        )
        self.base_url = str(_get_with_env(kwargs, "if_llm_verifier_base_url", "IF_LLM_VERIFIER_BASE_URL", "") or "")
        base_urls_value = _get_with_env(
            kwargs,
            "if_llm_verifier_base_urls",
            "IF_LLM_VERIFIER_BASE_URLS",
            "",
        )
        self.base_urls = _split_csv(base_urls_value)
        if self.base_url and self.base_url not in self.base_urls:
            self.base_urls.insert(0, self.base_url)
        if not self.base_urls and self.base_url:
            self.base_urls = [self.base_url]
        self.temperature = float(
            _get_with_env(kwargs, "if_llm_verifier_temperature", "IF_LLM_VERIFIER_TEMPERATURE", 0.0)
        )
        self.top_p = float(_get_with_env(kwargs, "if_llm_verifier_top_p", "IF_LLM_VERIFIER_TOP_P", 1.0))
        self.max_tokens = int(_get_with_env(kwargs, "if_llm_verifier_max_tokens", "IF_LLM_VERIFIER_MAX_TOKENS", 8192))
        self.omit_max_tokens = _as_bool(
            _get_with_env(kwargs, "if_llm_verifier_omit_max_tokens", "IF_LLM_VERIFIER_OMIT_MAX_TOKENS", False)
        )
        enable_thinking = str(
            _get_with_env(
                kwargs,
                "if_llm_verifier_enable_thinking",
                "IF_LLM_VERIFIER_ENABLE_THINKING",
                "",
            )
            or ""
        ).strip()
        self.enable_thinking = _as_bool(enable_thinking) if enable_thinking else None
        reasoning_effort = str(
            _get_with_env(kwargs, "if_llm_verifier_reasoning_effort", "IF_LLM_VERIFIER_REASONING_EFFORT", "") or ""
        ).strip()
        self.reasoning_effort = reasoning_effort if reasoning_effort in {"low", "medium", "high"} else ""
        self.timeout = float(_get_with_env(kwargs, "if_llm_verifier_timeout", "IF_LLM_VERIFIER_TIMEOUT", 120.0))
        self.max_retries = int(_get_with_env(kwargs, "if_llm_verifier_max_retries", "IF_LLM_VERIFIER_MAX_RETRIES", 2))
        self.response_format = _as_bool(
            _get_with_env(kwargs, "if_llm_verifier_response_format", "IF_LLM_VERIFIER_RESPONSE_FORMAT", False)
        )
        self.require_think_end = _as_bool(os.getenv("IF_REQUIRE_THINK_END_FOR_REWARD"), False)

        try:
            self.reward_fn_key = config.data.get("reward_fn_key", "data_source")
        except Exception:  # noqa: BLE001
            self.reward_fn_key = "data_source"

        logger.warning(
            "IFLLMVerifierRewardManager: model=%s endpoints=%s threshold=%s bonus=%s "
            "response_format=%s omit_max_tokens=%s enable_thinking=%s reasoning_effort=%s",
            self.model,
            ",".join(self.base_urls) or f"router:{self.reward_router_address}",
            self.threshold,
            self.bonus,
            self.response_format,
            self.omit_max_tokens,
            self.enable_thinking if self.enable_thinking is not None else "<default>",
            self.reasoning_effort or "<default>",
        )

    def _endpoint_url(self, prompt: str = "", response: str = "") -> str | None:
        if self.base_urls:
            if len(self.base_urls) == 1:
                return _chat_completions_url(self.base_urls[0])
            key = f"{prompt}\0{response}".encode("utf-8", errors="ignore")
            endpoint_idx = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(self.base_urls)
            return _chat_completions_url(self.base_urls[endpoint_idx])
        if self.reward_router_address:
            return f"http://{self.reward_router_address}/v1/chat/completions"
        return None

    async def _judge(self, prompt: str, response: str) -> tuple[int | None, str | None, str | None]:
        endpoint = self._endpoint_url(prompt, response)
        if not endpoint:
            return None, None, "missing IF_LLM_VERIFIER_BASE_URL(S) and reward_router_address"

        judge_prompt = DEFAULT_JUDGE_PROMPT.format(prompt=prompt, response=response)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if not self.omit_max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.response_format:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        raw_judgment = None
        for _ in range(max(self.max_retries, 0) + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(endpoint, json=payload) as resp:
                        body = await resp.text()
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}: {body[:1000]}")
                        data = json.loads(body)
                message = data["choices"][0].get("message", {})
                raw_judgment = _message_final_content(message)
                if not raw_judgment:
                    raise ValueError("LLM verifier did not return final message.content")
                return extract_judge_score(raw_judgment), raw_judgment, None
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        return None, raw_judgment, last_error

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        verifier_prompt = data_item.non_tensor_batch.get("ppl_prompt", None)
        if verifier_prompt is None:
            raise KeyError("IFLLMVerifierRewardManager requires `ppl_prompt` so the LLM verifier sees x only.")
        prompt_text = _prompt_to_text(verifier_prompt)

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        constraint_score = await self.loop.run_in_executor(
            None, lambda: score_ifeval(response_str, ground_truth, require_think_end=self.require_think_end)
        )
        base_reward = self.verification_reward * float(constraint_score)

        judge_called = False
        judge_score = None
        raw_judgment = None
        judge_error = None
        bonus = 0.0
        if constraint_score > 0.0:
            judge_called = True
            judge_response = remove_thinking_section(response_str, require_think_end=False)
            if judge_response is None:
                judge_response = response_str
            judge_score, raw_judgment, judge_error = await self._judge(prompt_text, judge_response)
            if judge_score is not None and judge_score >= self.threshold:
                bonus = self.bonus

        reward = base_reward + bonus
        return {
            "reward_score": float(reward),
            "reward_extra_info": {
                "acc": float(constraint_score),
                "verifier_score": float(constraint_score),
                "constraint_reward": float(base_reward),
                "llm_verifier_called": float(judge_called),
                "llm_verifier_score": float(judge_score if judge_score is not None else -1),
                "llm_verifier_pass": float(judge_score is not None and judge_score >= self.threshold),
                "llm_verifier_bonus": float(bonus),
                "llm_verifier_error": judge_error or "",
                "llm_verifier_prompt_source": "ppl_prompt",
                "llm_verifier_raw_judgment": (raw_judgment or "")[:500],
                "scaled_reward": float(reward),
                "valid_response_length": int(valid_response_length),
            },
        }
