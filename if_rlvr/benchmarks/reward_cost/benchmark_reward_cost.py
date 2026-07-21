#!/usr/bin/env python3
"""Benchmark IF-RLVR PPL and LLM-verifier reward inference on shared rollouts.

The timed region begins after all four vLLM replicas are healthy and after one
warmup request per replica.  Every benchmark consumes the same canonical JSONL
created by ``prepare_rollout_step.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import aiohttp
from transformers import AutoConfig

try:
    # Package import, e.g. ``python -m if_rlvr.benchmarks.reward_cost...``.
    from .flops import ModelSpec, estimate_generation, estimate_ppl
except ImportError:  # pragma: no cover - exercised by direct-script execution
    # Direct execution used by the launcher in this directory.
    from flops import ModelSpec, estimate_generation, estimate_ppl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": float(max(values)),
    }


def _endpoint_url(base_url: str, route: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{route.lstrip('/')}"
    return f"{base}/v1/{route.lstrip('/')}"


def _route_index(row: dict[str, Any], endpoint_count: int) -> int:
    prompt = str(row.get("judge_prompt", ""))
    response = str(row.get("judge_response", row.get("response_text", "")))
    key = f"{prompt}\0{response}".encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % endpoint_count


def _coerce_token_ids(value: Any, field: str, row_id: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        result = [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_id!r} has invalid {field}") from exc
    if any(token_id < 0 for token_id in result):
        raise ValueError(f"Row {row_id!r} has a negative token id in {field}")
    return result


def _validate_rows(
    rows: list[dict[str, Any]],
    expected_responses: int | None,
    expected_prompts: int | None,
    rollouts_per_prompt: int | None,
) -> dict[str, Any]:
    if expected_responses is not None and len(rows) != expected_responses:
        raise ValueError(f"Expected {expected_responses} responses, found {len(rows)}")
    ids = [row.get("id", index) for index, row in enumerate(rows)]
    if len({str(value) for value in ids}) != len(ids):
        raise ValueError("Canonical row ids are not unique")

    uid_counts: dict[str, int] = {}
    for row in rows:
        uid = str(row.get("uid", ""))
        if uid:
            uid_counts[uid] = uid_counts.get(uid, 0) + 1
    if expected_prompts is not None and len(uid_counts) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} unique prompt uids, found {len(uid_counts)}")
    if rollouts_per_prompt is not None and uid_counts:
        invalid = {uid: count for uid, count in uid_counts.items() if count != rollouts_per_prompt}
        if invalid:
            preview = list(invalid.items())[:5]
            raise ValueError(f"Expected {rollouts_per_prompt} responses per uid; examples: {preview}")

    # Empty final answers are valid inputs to the training verifier (for
    # example, a reasoning-only response after thinking removal).  Require the
    # canonical fields, but do not reject an empty string.
    missing_judge = [
        row.get("id")
        for row in rows
        if "judge_prompt" not in row
        or row.get("judge_prompt") is None
        or "judge_response" not in row
        or row.get("judge_response") is None
    ]
    if missing_judge:
        raise ValueError(f"Rows missing judge_prompt/judge_response: {missing_judge[:5]}")

    missing_ppl = []
    sequence_lengths: list[int] = []
    ppl_eligible_count = 0
    for index, row in enumerate(rows):
        row_id = row.get("id", index)
        prefix = _coerce_token_ids(row.get("ppl_prefix_token_ids"), "ppl_prefix_token_ids", row_id)
        continuation = _coerce_token_ids(
            row.get("ppl_continuation_token_ids"), "ppl_continuation_token_ids", row_id
        )
        explicit_eligible = row.get("ppl_eligible")
        ppl_eligible = (
            bool(prefix and continuation) if explicit_eligible is None else bool(explicit_eligible)
        )
        if ppl_eligible and (not prefix or not continuation):
            missing_ppl.append(row_id)
        elif ppl_eligible:
            ppl_eligible_count += 1
            sequence_lengths.append(len(prefix) + len(continuation))
    if missing_ppl:
        raise ValueError(f"Rows missing PPL prefix/continuation tokens: {missing_ppl[:5]}")

    return {
        "response_count": len(rows),
        "unique_prompt_count": len(uid_counts),
        "rollouts_per_prompt_values": sorted(set(uid_counts.values())),
        "ppl_eligible_response_count": ppl_eligible_count,
        "ppl_ineligible_response_count": len(rows) - ppl_eligible_count,
        "ppl_sequence_tokens": _summary([float(value) for value in sequence_lengths]),
    }


def _load_model_spec(model: str, trust_remote_code: bool, local_files_only: bool) -> ModelSpec:
    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return ModelSpec.from_config(config.to_dict())


def _extract_usage(data: dict[str, Any]) -> tuple[int, int, int, bool]:
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    cached_tokens_reported = "cached_tokens" in prompt_details
    return prompt_tokens, completion_tokens, cached_tokens, cached_tokens_reported


def _extract_reasoning_tokens(data: dict[str, Any]) -> int | None:
    usage = data.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" not in completion_details:
        return None
    return int(completion_details.get("reasoning_tokens", 0) or 0)


def _extract_logprob(row: Any, token_id: int) -> float | None:
    if row is None:
        return None
    if isinstance(row, (int, float)):
        return float(row)
    if isinstance(row, list):
        values = [_extract_logprob(item, token_id) for item in row]
        values = [value for value in values if value is not None]
        return max(values) if values else None
    if not isinstance(row, dict):
        return None

    direct = row.get(str(token_id), row.get(token_id))
    if isinstance(direct, (int, float)):
        return float(direct)
    if isinstance(direct, dict) and direct.get("logprob") is not None:
        return float(direct["logprob"])

    for value in row.values():
        if isinstance(value, dict):
            value_token_id = value.get("token_id")
            if value_token_id is not None and int(value_token_id) == token_id and value.get("logprob") is not None:
                return float(value["logprob"])
    return None


def _continuation_logprobs(
    prompt_logprobs: list[Any],
    prefix_ids: list[int],
    continuation_ids: list[int],
) -> tuple[list[float], str]:
    sequence_length = len(prefix_ids) + len(continuation_ids)
    candidates: list[tuple[str, int]] = []
    if len(prompt_logprobs) == sequence_length:
        candidates.append(("openai_prompt_index", len(prefix_ids)))
    if len(prompt_logprobs) == sequence_length - 1:
        candidates.append(("shifted_no_first_row", max(len(prefix_ids) - 1, 0)))
    # This is the slice used by the current IF-RLVR external-vLLM PPL path.
    candidates.append(("if_rlvr_legacy", max(len(prefix_ids) - 1, 0)))

    seen: set[tuple[str, int]] = set()
    for alignment, start in candidates:
        if (alignment, start) in seen:
            continue
        seen.add((alignment, start))
        rows = prompt_logprobs[start : start + len(continuation_ids)]
        if len(rows) != len(continuation_ids):
            continue
        values = [_extract_logprob(row, token_id) for row, token_id in zip(rows, continuation_ids, strict=True)]
        if all(value is not None for value in values):
            return [float(value) for value in values], alignment
    raise ValueError(
        "Could not align vLLM prompt_logprobs to continuation tokens: "
        f"rows={len(prompt_logprobs)} prefix={len(prefix_ids)} continuation={len(continuation_ids)}"
    )


@dataclass
class EndpointState:
    index: int
    base_url: str
    semaphore: asyncio.Semaphore
    first_request_start: float | None = None
    last_request_end: float | None = None
    request_count: int = 0

    @property
    def makespan_seconds(self) -> float:
        if self.first_request_start is None or self.last_request_end is None:
            return 0.0
        return max(self.last_request_end - self.first_request_start, 0.0)


@dataclass
class RequestKind:
    name: str
    route: str
    make_payload: Callable[[dict[str, Any]], dict[str, Any]]
    parse_response: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


async def _execute_request(
    row: dict[str, Any],
    state: EndpointState,
    session: aiohttp.ClientSession,
    kind: RequestKind,
    timeout_seconds: float,
    max_retries: int,
    timed: bool,
) -> dict[str, Any]:
    async with state.semaphore:
        start = time.perf_counter()
        if timed:
            if state.first_request_start is None:
                state.first_request_start = start
            state.request_count += 1

        attempts = 0
        last_error = ""
        status_code: int | None = None
        response_data: dict[str, Any] | None = None
        url = _endpoint_url(state.base_url, kind.route)
        for attempt in range(max(max_retries, 0) + 1):
            attempts = attempt + 1
            try:
                payload = kind.make_payload(row)
                timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                async with session.post(url, json=payload, timeout=timeout) as response:
                    status_code = response.status
                    body = await response.text()
                if status_code >= 400:
                    raise RuntimeError(f"HTTP {status_code}: {body[:1000]}")
                response_data = json.loads(body)
                parsed = kind.parse_response(row, response_data)
                end = time.perf_counter()
                if timed:
                    state.last_request_end = end
                return {
                    "id": row.get("id"),
                    "uid": row.get("uid", ""),
                    "endpoint_index": state.index,
                    "endpoint": state.base_url,
                    "success": True,
                    "attempts": attempts,
                    "http_status": status_code,
                    "latency_seconds": end - start,
                    "error": "",
                    **parsed,
                }
            except Exception as exc:  # noqa: BLE001 - preserve per-row failures
                last_error = f"{type(exc).__name__}: {exc}"

        end = time.perf_counter()
        if timed:
            state.last_request_end = end
        prompt_tokens, completion_tokens, cached_tokens, cached_tokens_reported = _extract_usage(
            response_data or {}
        )
        return {
            "id": row.get("id"),
            "uid": row.get("uid", ""),
            "endpoint_index": state.index,
            "endpoint": state.base_url,
            "success": False,
            "attempts": attempts,
            "http_status": status_code,
            "latency_seconds": end - start,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": _extract_reasoning_tokens(response_data or {}),
            "cached_prompt_tokens": cached_tokens,
            "cached_prompt_tokens_reported": cached_tokens_reported,
            "error": last_error,
        }


async def _run_requests(
    rows: list[dict[str, Any]],
    endpoints: list[str],
    kind: RequestKind,
    concurrency_per_endpoint: int,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("No rows selected for the benchmark")
    states = [
        EndpointState(index=index, base_url=base_url, semaphore=asyncio.Semaphore(concurrency_per_endpoint))
        for index, base_url in enumerate(endpoints)
    ]
    connectors = [aiohttp.TCPConnector(limit=max(concurrency_per_endpoint, 1)) for _ in endpoints]
    sessions = [aiohttp.ClientSession(connector=connector) for connector in connectors]
    try:
        warmup_started = time.perf_counter()
        warmups = await asyncio.gather(
            *[
                _execute_request(
                    rows[index % len(rows)],
                    states[index],
                    sessions[index],
                    kind,
                    timeout_seconds,
                    max_retries=0,
                    timed=False,
                )
                for index in range(len(endpoints))
            ]
        )
        warmup_seconds = time.perf_counter() - warmup_started
        warmup_errors = [result for result in warmups if not result["success"]]
        if warmup_errors:
            raise RuntimeError(f"Warmup failed: {warmup_errors}")

        wall_started = time.perf_counter()
        tasks = []
        for row in rows:
            endpoint_index = _route_index(row, len(endpoints))
            tasks.append(
                asyncio.create_task(
                    _execute_request(
                        row,
                        states[endpoint_index],
                        sessions[endpoint_index],
                        kind,
                        timeout_seconds,
                        max_retries,
                        timed=True,
                    )
                )
            )
        results = await asyncio.gather(*tasks)
        wall_seconds = time.perf_counter() - wall_started
    finally:
        await asyncio.gather(*[session.close() for session in sessions])

    endpoint_metrics = [
        {
            "index": state.index,
            "base_url": state.base_url,
            "request_count": state.request_count,
            "makespan_seconds": state.makespan_seconds,
        }
        for state in states
    ]
    replica_makespan_seconds = sum(item["makespan_seconds"] for item in endpoint_metrics)
    allocated_gpu_seconds = wall_seconds * len(endpoints)
    return results, {
        "warmup_seconds": warmup_seconds,
        "wall_seconds_4gpu": wall_seconds,
        "endpoint_metrics": endpoint_metrics,
        "one_gpu_equivalent_replica_seconds": replica_makespan_seconds,
        "one_gpu_equivalent_allocated_seconds": allocated_gpu_seconds,
        "replica_span_efficiency": (
            replica_makespan_seconds / allocated_gpu_seconds if allocated_gpu_seconds > 0 else 0.0
        ),
    }


def _make_ppl_kind(model: str) -> RequestKind:
    def make_payload(row: dict[str, Any]) -> dict[str, Any]:
        row_id = row.get("id")
        prefix = _coerce_token_ids(row.get("ppl_prefix_token_ids"), "ppl_prefix_token_ids", row_id)
        continuation = _coerce_token_ids(
            row.get("ppl_continuation_token_ids"), "ppl_continuation_token_ids", row_id
        )
        if not prefix or not continuation:
            raise ValueError("PPL requires non-empty prefix and continuation token ids")
        return {
            "model": model,
            "prompt": prefix + continuation,
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 1,
            "add_special_tokens": False,
        }

    def parse_response(row: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        row_id = row.get("id")
        prefix = _coerce_token_ids(row.get("ppl_prefix_token_ids"), "ppl_prefix_token_ids", row_id)
        continuation = _coerce_token_ids(
            row.get("ppl_continuation_token_ids"), "ppl_continuation_token_ids", row_id
        )
        choice = data["choices"][0]
        logprob_payload = choice.get("logprobs") or {}
        prompt_logprobs = logprob_payload.get("prompt_logprobs") or choice.get("prompt_logprobs") or []
        token_logprobs, alignment = _continuation_logprobs(prompt_logprobs, prefix, continuation)
        nll = -sum(token_logprobs)
        mean_nll = nll / len(token_logprobs)
        prompt_tokens, completion_tokens, cached_tokens, cached_tokens_reported = _extract_usage(data)
        return {
            "prefix_tokens": len(prefix),
            "scored_tokens": len(continuation),
            "sequence_tokens": len(prefix) + len(continuation),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached_tokens,
            "cached_prompt_tokens_reported": cached_tokens_reported,
            "nll": nll,
            "mean_nll": mean_nll,
            "ppl": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
            "prompt_logprob_alignment": alignment,
        }

    return RequestKind(name="ppl", route="completions", make_payload=make_payload, parse_response=parse_response)


def _load_judge_helpers():
    try:
        from if_rlvr.if_llm_verifier_reward_manager import (  # type: ignore
            DEFAULT_JUDGE_PROMPT,
            _message_final_content,
            extract_judge_score,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import the training verifier prompt/parser. Set PYTHONPATH to the if-rlvr repository root."
        ) from exc
    return DEFAULT_JUDGE_PROMPT, _message_final_content, extract_judge_score


def _make_verifier_kind(
    model: str,
    max_tokens: int,
    omit_max_tokens: bool,
    enable_thinking: bool | None,
    reasoning_effort: str,
    response_format: bool,
    temperature: float,
    top_p: float,
) -> RequestKind:
    judge_template, final_content, extract_score = _load_judge_helpers()

    def make_payload(row: dict[str, Any]) -> dict[str, Any]:
        judge_prompt = judge_template.format(prompt=row["judge_prompt"], response=row["judge_response"])
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": temperature,
            "top_p": top_p,
        }
        if not omit_max_tokens:
            payload["max_tokens"] = max_tokens
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def parse_response(row: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        message = data["choices"][0].get("message", {})
        raw_judgment = final_content(message) or ""
        score: int | None = None
        judge_parse_error = ""
        if not raw_judgment:
            judge_parse_error = "Verifier did not return final message.content"
        else:
            try:
                score = extract_score(raw_judgment)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                # A malformed score is model output, not an inference failure.
                # Keep its measured latency, token usage, and FLOPs while making
                # the parse failure explicit in results and the report.
                judge_parse_error = f"{type(exc).__name__}: {exc}"
        prompt_tokens, completion_tokens, cached_tokens, cached_tokens_reported = _extract_usage(data)
        if prompt_tokens <= 0 or completion_tokens <= 0:
            raise ValueError(
                "Verifier response is missing nonzero API token usage; refusing to record zero FLOPs"
            )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": _extract_reasoning_tokens(data),
            "cached_prompt_tokens": cached_tokens,
            "cached_prompt_tokens_reported": cached_tokens_reported,
            "judge_score": score,
            "judge_parse_error": judge_parse_error,
            "raw_judgment": raw_judgment,
        }

    return RequestKind(
        name="verifier",
        route="chat/completions",
        make_payload=make_payload,
        parse_response=parse_response,
    )


def _attach_flops(results: list[dict[str, Any]], spec: ModelSpec, benchmark_type: str) -> None:
    for result in results:
        if not result.get("success"):
            continue
        if benchmark_type == "ppl":
            components = estimate_ppl(spec, int(result["sequence_tokens"]))
            full_components = components
        else:
            prompt_tokens = int(result.get("prompt_tokens", 0))
            completion_tokens = int(result.get("completion_tokens", 0))
            cached_tokens = int(result.get("cached_prompt_tokens", 0))
            components = estimate_generation(spec, prompt_tokens, completion_tokens, cached_tokens)
            full_components = estimate_generation(spec, prompt_tokens, completion_tokens, 0)
        result["estimated_logical_flops"] = components
        result["estimated_full_no_cache_flops"] = full_components


def _aggregate_metrics(
    args: argparse.Namespace,
    selected_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    timing: dict[str, Any],
    spec: ModelSpec,
) -> dict[str, Any]:
    successes = [result for result in results if result.get("success")]
    failures = [result for result in results if not result.get("success")]
    total_flops = sum(int(result["estimated_logical_flops"]["total"]) for result in successes)
    total_full_flops = sum(int(result["estimated_full_no_cache_flops"]["total"]) for result in successes)
    replica_seconds = float(timing["one_gpu_equivalent_replica_seconds"])
    wall_seconds = float(timing["wall_seconds_4gpu"])

    prompt_tokens = [float(result.get("prompt_tokens", 0)) for result in successes]
    completion_tokens = [float(result.get("completion_tokens", 0)) for result in successes]
    reasoning_tokens = [
        float(result["reasoning_tokens"])
        for result in successes
        if result.get("reasoning_tokens") is not None
    ]
    sequence_tokens = [float(result.get("sequence_tokens", 0)) for result in successes if "sequence_tokens" in result]
    scored_tokens = [float(result.get("scored_tokens", 0)) for result in successes if "scored_tokens" in result]
    latencies = [float(result["latency_seconds"]) for result in results]
    cached_prompt_tokens = sum(int(result.get("cached_prompt_tokens", 0)) for result in successes)
    cached_usage_detail_count = sum(
        bool(result.get("cached_prompt_tokens_reported", False)) for result in successes
    )
    total_attempts = sum(int(result.get("attempts", 1)) for result in results)
    retry_count = sum(max(int(result.get("attempts", 1)) - 1, 0) for result in results)
    judge_parse_errors = [
        str(result["judge_parse_error"])
        for result in successes
        if result.get("judge_parse_error")
    ]

    if (
        args.command == "verifier"
        and args.prefix_caching == "enabled"
        and cached_usage_detail_count != len(successes)
    ):
        raise ValueError(
            "Prefix caching is enabled, but cached-token usage details are missing from one or more "
            "responses. Start vLLM with --enable-prompt-tokens-details."
        )

    metrics = {
        "schema_version": 1,
        "method": args.method,
        "benchmark_type": args.command,
        "model": args.model,
        "hardware": args.hardware,
        "model_spec": spec.to_dict(),
        "input_file": str(Path(args.input).resolve()),
        "input_sha256": _file_sha256(Path(args.input)),
        "input_response_count": len(all_rows),
        "selected_response_count": len(selected_rows),
        "skipped_response_count": len(all_rows) - len(selected_rows),
        "success_count": len(successes),
        "error_count": len(failures),
        "total_http_attempts": total_attempts,
        "retry_count": retry_count,
        "judge_parse_error_count": len(judge_parse_errors),
        "judge_mode": getattr(args, "judge_mode", "all"),
        "replica_count": len(args.endpoints),
        "tensor_parallel_size": 1,
        "effective_data_parallel_size": len(args.endpoints),
        "client_concurrency_per_endpoint": args.concurrency_per_endpoint,
        "server_config": {
            "gpu_memory_utilization": args.server_gpu_memory_utilization,
            "max_model_len": args.server_max_model_len,
            "max_num_batched_tokens": args.server_max_num_batched_tokens,
            "max_num_seqs": args.server_max_num_seqs,
        },
        "prefix_caching_expected": args.prefix_caching,
        "http_session_mode": "persistent-per-endpoint",
        "server_startup_and_model_load_excluded": True,
        "estimated_logical_flops": total_flops,
        "estimated_logical_petaflops": total_flops / 1e15,
        "estimated_full_no_cache_flops": total_full_flops,
        "estimated_full_no_cache_petaflops": total_full_flops / 1e15,
        "logical_tflops_per_replica_second": (
            total_flops / replica_seconds / 1e12 if replica_seconds else 0.0
        ),
        "requests_per_4gpu_wall_second": len(successes) / wall_seconds if wall_seconds else 0.0,
        "total_prompt_tokens": int(sum(prompt_tokens)),
        "total_completion_tokens": int(sum(completion_tokens)),
        "total_reasoning_tokens": int(sum(reasoning_tokens)) if reasoning_tokens else None,
        "reasoning_token_usage_available_count": len(reasoning_tokens),
        "total_cached_prompt_tokens": cached_prompt_tokens,
        "cached_prompt_token_usage_available_count": cached_usage_detail_count,
        "prompt_tokens": _summary(prompt_tokens),
        "completion_tokens": _summary(completion_tokens),
        "reasoning_tokens": _summary(reasoning_tokens),
        "sequence_tokens": _summary(sequence_tokens),
        "scored_tokens": _summary(scored_tokens),
        "request_latency_seconds": _summary(latencies),
        "errors_preview": [result.get("error", "") for result in failures[:10]],
        "judge_parse_errors_preview": judge_parse_errors[:10],
        "timing": timing,
        "one_gpu_equivalent_note": (
            "replica estimate is the sum of first-request-to-last-response makespans for the independent TP=1 "
            "replicas; it includes server queue and HTTP/CPU overhead and is a projected serial cost, not measured "
            "GPU kernel-active time. Allocated estimate is 4-GPU wall seconds multiplied by four. Neither is a "
            "direct single-GPU rerun."
        ),
        "flops_note": (
            "Analytical logical matmul FLOPs with FMA=2; active MoE experts only. Norm/RoPE/softmax, memory IO, "
            "kernel padding, discarded chunked-prefill logits and HTTP/CPU work are excluded. These are useful-model "
            "FLOPs, not hardware performance-counter FLOPs."
        ),
        "retry_note": (
            "Default launcher retries are disabled. If retry_count is nonzero, elapsed time contains failed attempts "
            "while logical FLOPs cover only the final successful responses."
        ),
    }
    return metrics


def _write_run(output_dir: Path, results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: str(item.get("id"))):
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=True) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def _parse_endpoints(value: str) -> list[str]:
    endpoints = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
    if not endpoints:
        raise argparse.ArgumentTypeError("At least one endpoint is required")
    return endpoints


def _optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"none", "default", "auto"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true, false, or none; got {value!r}")


def _run_benchmark(args: argparse.Namespace, kind: RequestKind) -> int:
    input_path = Path(args.input)
    all_rows = _read_jsonl(input_path)
    _validate_rows(all_rows, args.expected_responses, args.expected_prompts, args.rollouts_per_prompt)
    if args.command == "ppl":
        selected_rows = []
        for index, row in enumerate(all_rows):
            row_id = row.get("id", index)
            prefix = _coerce_token_ids(
                row.get("ppl_prefix_token_ids"), "ppl_prefix_token_ids", row_id
            )
            continuation = _coerce_token_ids(
                row.get("ppl_continuation_token_ids"), "ppl_continuation_token_ids", row_id
            )
            explicit_eligible = row.get("ppl_eligible")
            eligible = bool(prefix and continuation) if explicit_eligible is None else bool(explicit_eligible)
            if eligible:
                selected_rows.append(row)
    elif args.command == "verifier" and args.judge_mode == "constraint-positive":
        missing = [row.get("id") for row in all_rows if "constraint_score" not in row]
        if missing:
            raise ValueError(f"constraint-positive mode requires constraint_score; missing rows: {missing[:5]}")
        selected_rows = [row for row in all_rows if float(row.get("constraint_score", 0.0)) > 0.0]
    else:
        selected_rows = all_rows

    spec = _load_model_spec(args.model, args.trust_remote_code, args.local_files_only)
    if selected_rows:
        results, timing = asyncio.run(
            _run_requests(
                selected_rows,
                args.endpoints,
                kind,
                args.concurrency_per_endpoint,
                args.timeout,
                args.max_retries,
            )
        )
    else:
        timing = {
            "warmup_seconds": 0.0,
            "wall_seconds_4gpu": 0.0,
            "endpoint_metrics": [
                {
                    "index": index,
                    "base_url": endpoint,
                    "request_count": 0,
                    "makespan_seconds": 0.0,
                }
                for index, endpoint in enumerate(args.endpoints)
            ],
            "one_gpu_equivalent_replica_seconds": 0.0,
            "one_gpu_equivalent_allocated_seconds": 0.0,
            "replica_span_efficiency": 0.0,
        }
        results = []
    _attach_flops(results, spec, args.command)
    metrics = _aggregate_metrics(args, selected_rows, all_rows, results, timing, spec)
    _write_run(Path(args.output), results, metrics)
    print(json.dumps({"output": str(Path(args.output).resolve()), **metrics["timing"]}, indent=2))
    if metrics["error_count"] and not args.allow_errors:
        print(f"ERROR: {metrics['error_count']} requests failed; see {args.output}/metrics.json")
        return 2
    return 0


def command_validate(args: argparse.Namespace) -> int:
    path = Path(args.input)
    rows = _read_jsonl(path)
    summary = _validate_rows(rows, args.expected_responses, args.expected_prompts, args.rollouts_per_prompt)
    summary["input"] = str(path.resolve())
    summary["sha256"] = _file_sha256(path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_ppl(args: argparse.Namespace) -> int:
    return _run_benchmark(args, _make_ppl_kind(args.model))


def command_verifier(args: argparse.Namespace) -> int:
    kind = _make_verifier_kind(
        model=args.model,
        max_tokens=args.max_tokens,
        omit_max_tokens=args.omit_max_tokens,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        response_format=args.response_format,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return _run_benchmark(args, kind)


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    timing = metrics["timing"]
    server_config = metrics.get("server_config") or {}
    return {
        "method": metrics["method"],
        "model": metrics["model"],
        "hardware": metrics.get("hardware", ""),
        "input_responses": metrics["input_response_count"],
        "selected_requests": metrics["selected_response_count"],
        "successful_requests": metrics["success_count"],
        "errors": metrics["error_count"],
        "judge_parse_errors": metrics.get("judge_parse_error_count", 0),
        "retries": metrics.get("retry_count", 0),
        "client_concurrency": metrics.get("client_concurrency_per_endpoint"),
        "max_batched_tokens": server_config.get("max_num_batched_tokens"),
        "max_num_seqs": server_config.get("max_num_seqs"),
        "gpu_memory_utilization": server_config.get("gpu_memory_utilization"),
        "4gpu_wall_s": timing["wall_seconds_4gpu"],
        "1gpu_eq_replica_s": timing["one_gpu_equivalent_replica_seconds"],
        "1gpu_eq_allocated_s": timing["one_gpu_equivalent_allocated_seconds"],
        "replica_span_efficiency": timing["replica_span_efficiency"],
        "requests_per_s": metrics["requests_per_4gpu_wall_second"],
        "prompt_tokens": metrics["total_prompt_tokens"],
        "completion_tokens": metrics["total_completion_tokens"],
        "logical_PFLOPs": metrics["estimated_logical_petaflops"],
        "logical_TFLOPs_per_replica_s": metrics["logical_tflops_per_replica_second"],
    }


def command_report(args: argparse.Namespace) -> int:
    metric_paths: list[Path] = []
    for raw_path in args.metrics:
        path = Path(raw_path)
        if path.is_dir():
            path = path / "metrics.json"
        metric_paths.append(path)
    payloads = []
    for path in metric_paths:
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    rows = [_metric_row(payload) for payload in payloads]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": payloads, "comparison": rows}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "method",
        "model",
        "hardware",
        "input responses",
        "selected requests",
        "successful requests",
        "errors",
        "judge parse errors",
        "retries",
        "client concurrency/GPU",
        "max batched tokens",
        "max sequences",
        "GPU memory utilization",
        "4-GPU wall (s)",
        "1-GPU eq replica-sum (s)",
        "1-GPU eq allocated (s)",
        "replica span / allocation",
        "req/s",
        "prompt tokens",
        "completion tokens",
        "logical PFLOPs",
        "logical TFLOP/replica-s",
    ]
    markdown_lines = [
        "# Reward cost comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["model"]),
                    str(row["hardware"]),
                    str(row["input_responses"]),
                    str(row["selected_requests"]),
                    str(row["successful_requests"]),
                    str(row["errors"]),
                    str(row["judge_parse_errors"]),
                    str(row["retries"]),
                    str(row["client_concurrency"]),
                    str(row["max_batched_tokens"]),
                    str(row["max_num_seqs"]),
                    str(row["gpu_memory_utilization"]),
                    f"{row['4gpu_wall_s']:.3f}",
                    f"{row['1gpu_eq_replica_s']:.3f}",
                    f"{row['1gpu_eq_allocated_s']:.3f}",
                    f"{row['replica_span_efficiency']:.4f}",
                    f"{row['requests_per_s']:.3f}",
                    str(row["prompt_tokens"]),
                    str(row["completion_tokens"]),
                    f"{row['logical_PFLOPs']:.6f}",
                    f"{row['logical_TFLOPs_per_replica_s']:.3f}",
                ]
            )
            + " |"
        )
    markdown_lines.extend(
        [
            "",
            "`1-GPU eq replica-sum` is the sum of the four first-request-to-last-response replica makespans.",
            "It includes server queue and HTTP/CPU overhead; it is not measured GPU kernel-active time.",
            "`1-GPU eq allocated` is four times the 4-GPU wall time and includes tail idle time.",
            "Both are normalized costs, not a direct one-GPU rerun measurement.",
            "",
            "FLOPs are analytical useful-model matmul FLOPs (FMA=2), not hardware-counter FLOPs; server load and warmup are excluded from wall time.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print("\n".join(markdown_lines))
    return 0


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Canonical rollout JSONL")
    parser.add_argument("--expected-responses", type=int, default=8192)
    parser.add_argument("--expected-prompts", type=int, default=1024)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    _add_corpus_args(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", default="", help="Hardware label recorded in metrics/report")
    parser.add_argument("--endpoints", required=True, type=_parse_endpoints)
    parser.add_argument("--concurrency-per-endpoint", type=int, default=128)
    parser.add_argument("--server-gpu-memory-utilization", type=float)
    parser.add_argument("--server-max-model-len", type=int)
    parser.add_argument("--server-max-num-batched-tokens", type=int)
    parser.add_argument("--server-max-num-seqs", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument(
        "--prefix-caching",
        choices=("disabled", "enabled", "unknown"),
        default="disabled",
        help="Launcher state recorded in the report; cached tokens from API usage still drive FLOPs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate the canonical 1024 x 8 corpus")
    _add_corpus_args(validate)
    validate.set_defaults(func=command_validate)

    ppl = subparsers.add_parser("ppl", help="Qwen3-4B p(y|x) prompt-logprob benchmark")
    _add_benchmark_args(ppl)
    ppl.set_defaults(func=command_ppl)

    verifier = subparsers.add_parser("verifier", help="G-Eval-style LLM verifier benchmark")
    _add_benchmark_args(verifier)
    verifier.add_argument("--judge-mode", choices=("all", "constraint-positive"), default="all")
    verifier.add_argument("--max-tokens", type=int, default=2048)
    verifier.add_argument("--omit-max-tokens", action=argparse.BooleanOptionalAction, default=False)
    verifier.add_argument("--enable-thinking", type=_optional_bool, default=None)
    verifier.add_argument("--reasoning-effort", choices=("", "low", "medium", "high"), default="")
    verifier.add_argument("--response-format", action=argparse.BooleanOptionalAction, default=True)
    verifier.add_argument("--temperature", type=float, default=0.0)
    verifier.add_argument("--top-p", type=float, default=1.0)
    verifier.set_defaults(func=command_verifier)

    report = subparsers.add_parser("report", help="Merge metrics.json files into JSON/CSV/Markdown")
    report.add_argument("--metrics", nargs="+", required=True, help="Run directories or metrics.json files")
    report.add_argument("--output", required=True)
    report.set_defaults(func=command_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
