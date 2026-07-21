#!/usr/bin/env python3
"""Export one VERL rollout-skip step as deterministic reward-benchmark JSONL.

The input is a ``genstep_*`` directory containing ``new_batch.dp`` and
``gen_batch.dp``.  The output intentionally keeps only unpadded response token
ids and the fields needed to replay the same 8,192 trajectories through the
PPL and LLM-verifier reward paths.

The canonical PPL field names are ``ppl_prefix_token_ids`` and
``ppl_continuation_token_ids``.  They are populated from the rollout fields
``ppl_x_prompt_ids`` and ``ppl_y_final_answer_ids`` whenever those fields are
available, so the benchmark reuses the exact tokens seen by training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "if-rlvr-reward-cost-rollout-v1"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"
DEFAULT_EXPECTED_ROWS = 8192
DEFAULT_EXPECTED_UIDS = 1024
DEFAULT_ROLLOUTS_PER_UID = 8
DEFAULT_MAX_PROMPT_LENGTH = 16384
DEFAULT_NLTK_DATA = "/NHNHOME/WORKSPACE/26msit001_A/IFIF/IFBench/.nltk_data"
_MISSING = object()


def _load_dumped_data(step_dir: Path) -> tuple[Any, Any, str]:
    """Load a rollout dump, using the public helper when this VERL has it."""

    try:
        from verl.utils.rollout_skip import read_dumped_data
    except (ImportError, AttributeError):
        read_dumped_data = None

    if read_dumped_data is not None:
        dumped = read_dumped_data(step_dir)
        try:
            return dumped["new_batch"], dumped["gen_batch"], "verl.utils.rollout_skip.read_dumped_data"
        except (KeyError, TypeError) as exc:
            raise ValueError("read_dumped_data did not return new_batch/gen_batch") from exc

    try:
        from verl import DataProto
    except ImportError as exc:
        raise RuntimeError(
            "Could not import VERL. Run with the if-rlvr repository on PYTHONPATH "
            "and its verl Python environment."
        ) from exc

    new_batch_path = step_dir / "new_batch.dp"
    gen_batch_path = step_dir / "gen_batch.dp"
    if not new_batch_path.is_file() or not gen_batch_path.is_file():
        raise FileNotFoundError(f"Missing new_batch.dp or gen_batch.dp under {step_dir}")
    return (
        DataProto.load_from_disk(new_batch_path),
        DataProto.load_from_disk(gen_batch_path),
        "verl.DataProto.load_from_disk",
    )


def _load_tokenizer(
    model_name_or_path: str,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to prepare canonical rollouts") from exc

    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


def _load_verifier_functions() -> tuple[Callable[..., str | None], Callable[..., float]]:
    try:
        from if_rlvr.ifeval_oi.verifier import remove_thinking_section, score_ifeval
    except ImportError as exc:
        raise RuntimeError(
            "Could not import if_rlvr.ifeval_oi.verifier. Put the if-rlvr repository root on PYTHONPATH."
        ) from exc
    return remove_thinking_section, score_ifeval


def _configure_nltk_data(path: Path) -> None:
    """Fail fast instead of silently turning tokenizer-dependent constraints into zeroes."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"NLTK data directory does not exist: {resolved}")
    os.environ["NLTK_DATA"] = str(resolved)
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("nltk is required to reproduce training constraint scores") from exc
    if str(resolved) not in nltk.data.path:
        nltk.data.path.insert(0, str(resolved))
    missing = []
    for resource in ("tokenizers/punkt/english.pickle", "tokenizers/punkt_tab/english/"):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)
    if missing:
        raise RuntimeError(f"Required NLTK resources are missing under {resolved}: {missing}")


def _jsonable(value: Any) -> Any:
    """Convert tensor/numpy/object values into strict JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
        return _jsonable(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    return str(value)


def _prompt_to_text(raw_prompt: Any) -> str:
    """Match IFLLMVerifierRewardManager._prompt_to_text exactly."""

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
    except (TypeError, ValueError):
        return str(raw_prompt)


def _batch_size(proto: Any) -> int:
    batch = getattr(proto, "batch", None)
    if batch is not None:
        batch_size = getattr(batch, "batch_size", None)
        if batch_size is not None and len(batch_size) > 0:
            return int(batch_size[0])
        if isinstance(batch, Mapping) and batch:
            return len(next(iter(batch.values())))
        try:
            return len(batch)
        except TypeError:
            pass

    # RolloutSkip stores the repeated input ``new_batch`` as a non-tensor-only
    # DataProto in this VERL version.  DataProto.__len__ supports that layout,
    # but keep an explicit fallback for test doubles and version variants.
    try:
        size = len(proto)
    except TypeError:
        size = 0
    if size:
        return int(size)

    non_tensor_batch = getattr(proto, "non_tensor_batch", None) or {}
    if non_tensor_batch:
        return len(next(iter(non_tensor_batch.values())))
    return 0


def _has_batch_key(proto: Any, key: str) -> bool:
    if getattr(proto, "batch", None) is None:
        return False
    try:
        return key in proto.batch.keys()
    except AttributeError:
        return key in proto.batch


def _batch_value(proto: Any, key: str, index: int) -> Any:
    if not _has_batch_key(proto, key):
        return _MISSING
    return proto.batch[key][index]


def _non_tensor_value(proto: Any, key: str, index: int) -> Any:
    values = getattr(proto, "non_tensor_batch", {})
    if key not in values:
        return _MISSING
    value = values[key]
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        if _batch_size(proto) == 1:
            return value
        raise ValueError(f"Non-tensor field {key!r} is not row-aligned")


def _first_non_tensor_value(protos: Sequence[Any], key: str, index: int, default: Any = None) -> Any:
    for proto in protos:
        value = _non_tensor_value(proto, key, index)
        if value is not _MISSING and value is not None:
            return value
    return default


def _flat_list(value: Any, field: str, index: int) -> list[Any]:
    if value is _MISSING or value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Row {index}: {field} must be a token/mask sequence, not text")
    if not isinstance(value, (list, tuple)):
        try:
            value = list(value)
        except TypeError as exc:
            raise ValueError(f"Row {index}: {field} is not a sequence") from exc
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if any(isinstance(item, (list, tuple)) for item in value):
        raise ValueError(f"Row {index}: {field} must be one-dimensional")
    return list(value)


def _token_ids(value: Any, field: str, index: int) -> list[int]:
    values = _flat_list(value, field, index)
    try:
        token_ids = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {index}: {field} contains a non-integer token id") from exc
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"Row {index}: {field} contains a negative token id")
    return token_ids


def _mask_values(value: Any, field: str, index: int) -> list[bool]:
    values = _flat_list(value, field, index)
    return [bool(value) for value in values]


def _extract_response_token_ids(gen_batch: Any, index: int) -> list[int]:
    """Select generated tokens with response_mask and the response attention mask."""

    response_value = _batch_value(gen_batch, "responses", index)
    if response_value is _MISSING:
        raise KeyError("gen_batch is missing tensor field 'responses'")
    response_ids = _token_ids(response_value, "responses", index)
    width = len(response_ids)

    masks: list[list[bool]] = []
    response_mask_value = _batch_value(gen_batch, "response_mask", index)
    if response_mask_value is not _MISSING:
        response_mask = _mask_values(response_mask_value, "response_mask", index)
        if len(response_mask) != width:
            raise ValueError(
                f"Row {index}: response_mask width {len(response_mask)} != responses width {width}"
            )
        masks.append(response_mask)

    attention_mask_value = _batch_value(gen_batch, "attention_mask", index)
    if attention_mask_value is not _MISSING:
        attention_mask = _mask_values(attention_mask_value, "attention_mask", index)
        if len(attention_mask) < width:
            raise ValueError(
                f"Row {index}: attention_mask width {len(attention_mask)} < responses width {width}"
            )
        masks.append(attention_mask[-width:] if width else [])

    if not masks:
        raise KeyError("gen_batch needs response_mask or attention_mask to remove response padding exactly")
    valid_mask = [all(mask[position] for mask in masks) for position in range(width)]
    return [token_id for token_id, keep in zip(response_ids, valid_mask, strict=True) if keep]


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    try:
        return str(tokenizer.decode(list(token_ids), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(token_ids), **kwargs))


def _encode(tokenizer: Any, text: str) -> list[int]:
    return _token_ids(tokenizer.encode(text, add_special_tokens=False), "tokenizer.encode", -1)


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for position in range(len(sequence) - len(pattern) + 1):
        if list(sequence[position : position + len(pattern)]) == list(pattern):
            return position
    return None


def _strip_leading_whitespace_ids(tokenizer: Any, token_ids: list[int]) -> list[int]:
    if not token_ids:
        return []
    text = _decode(tokenizer, token_ids)
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped == text:
        return token_ids
    return _encode(tokenizer, stripped)


def _reconstruct_ppl_prefix(tokenizer: Any, ppl_prompt: Any, max_prompt_length: int, index: int) -> list[int]:
    messages = _jsonable(ppl_prompt)
    if isinstance(messages, Mapping):
        messages = [messages]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row {index}: ppl_prompt must be a non-empty chat message list")
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise TypeError(
            "The tokenizer chat template did not accept enable_thinking=False; "
            "use the Qwen/Qwen3-4B tokenizer required by this experiment."
        ) from exc
    token_ids = _token_ids(encoded, "apply_chat_template", index)
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be positive")
    return token_ids[-max_prompt_length:]


def _select_ppl_prefix(
    gen_batch: Any,
    tokenizer: Any,
    ppl_prompt: Any,
    max_prompt_length: int,
    index: int,
) -> tuple[list[int], str]:
    dumped = _non_tensor_value(gen_batch, "ppl_x_prompt_ids", index)
    dumped_ids = _token_ids(dumped, "ppl_x_prompt_ids", index) if dumped is not _MISSING else []
    if dumped_ids:
        return dumped_ids, "gen_batch.ppl_x_prompt_ids"
    return (
        _reconstruct_ppl_prefix(tokenizer, ppl_prompt, max_prompt_length, index),
        "reconstructed:ppl_prompt+chat_template(enable_thinking=false)",
    )


def _select_ppl_continuation(
    gen_batch: Any,
    tokenizer: Any,
    response_token_ids: list[int],
    index: int,
) -> tuple[list[int], str]:
    dumped = _non_tensor_value(gen_batch, "ppl_y_final_answer_ids", index)
    dumped_ids = _token_ids(dumped, "ppl_y_final_answer_ids", index) if dumped is not _MISSING else []
    if dumped_ids:
        return dumped_ids, "gen_batch.ppl_y_final_answer_ids"

    final_answer = _non_tensor_value(gen_batch, "if_final_answer_ids", index)
    final_answer_ids = (
        _token_ids(final_answer, "if_final_answer_ids", index) if final_answer is not _MISSING else []
    )
    if final_answer_ids:
        return final_answer_ids, "gen_batch.if_final_answer_ids"

    think_end_ids = _encode(tokenizer, "</think>")
    marker_position = _find_subsequence(response_token_ids, think_end_ids)
    if marker_position is not None:
        answer_ids = response_token_ids[marker_position + len(think_end_ids) :]
        answer_ids = _strip_leading_whitespace_ids(tokenizer, answer_ids)
        if answer_ids:
            return answer_ids, "derived:response_after_</think>"

    # Match the deferred ref-policy path: whitespace is stripped even when the
    # whole response is used as the fallback.  Empty/aborted rows remain in the
    # 8,192-row corpus but are ineligible for a model PPL call, exactly as
    # _build_ref_policy_scoring_batch filters empty prefix/continuation pairs.
    fallback_ids = _strip_leading_whitespace_ids(tokenizer, response_token_ids)
    if fallback_ids:
        return fallback_ids, "fallback:whole_response_stripped"
    return [], "ineligible:empty_or_whitespace_response"


def _extract_uid(new_batch: Any, gen_batch: Any, index: int) -> str:
    value = _first_non_tensor_value((new_batch, gen_batch), "uid", index, default="")
    value = _jsonable(value)
    return "" if value is None else str(value)


def _extract_ground_truth(new_batch: Any, gen_batch: Any, index: int) -> Any:
    reward_model = _first_non_tensor_value((new_batch, gen_batch), "reward_model", index, default=None)
    reward_model = _jsonable(reward_model)
    if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
        return reward_model["ground_truth"]
    direct = _first_non_tensor_value((new_batch, gen_batch), "ground_truth", index, default=_MISSING)
    if direct is not _MISSING and direct is not None:
        return _jsonable(direct)
    raise KeyError(f"Row {index}: reward_model.ground_truth is missing")


def _rollout_shape(
    new_batch: Any,
    gen_batch: Any,
    expected_rows: int,
    expected_unique_uids: int,
    expected_rollouts_per_uid: int,
    allow_count_mismatch: bool,
) -> tuple[dict[str, Any], list[str]]:
    new_size = _batch_size(new_batch)
    gen_size = _batch_size(gen_batch)
    if new_size != gen_size:
        raise ValueError(f"new_batch has {new_size} rows but gen_batch has {gen_size} rows")

    uids = [_extract_uid(new_batch, gen_batch, index) for index in range(gen_size)]
    uid_counts = Counter(uids)
    missing_uid_rows = uid_counts.pop("", 0)
    errors: list[str] = []
    if gen_size != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {gen_size}")
    if len(uid_counts) != expected_unique_uids:
        errors.append(f"expected {expected_unique_uids} non-empty unique uids, found {len(uid_counts)}")
    if missing_uid_rows:
        errors.append(f"{missing_uid_rows} rows have a missing/empty uid")
    bad_uid_counts = {uid: count for uid, count in uid_counts.items() if count != expected_rollouts_per_uid}
    if bad_uid_counts:
        preview = list(bad_uid_counts.items())[:5]
        errors.append(
            f"expected {expected_rollouts_per_uid} rows per uid; "
            f"{len(bad_uid_counts)} uids differ (examples: {preview})"
        )

    if errors and not allow_count_mismatch:
        raise ValueError("Rollout shape validation failed: " + "; ".join(errors))
    if errors:
        for error in errors:
            print(f"WARNING: {error}", file=sys.stderr, flush=True)

    count_histogram = Counter(uid_counts.values())
    validation = {
        "passed": not errors,
        "allow_count_mismatch": allow_count_mismatch,
        "expected": {
            "rows": expected_rows,
            "unique_uids": expected_unique_uids,
            "rollouts_per_uid": expected_rollouts_per_uid,
        },
        "observed": {
            "rows": gen_size,
            "unique_nonempty_uids": len(uid_counts),
            "missing_uid_rows": missing_uid_rows,
            "rollouts_per_uid_histogram": {
                str(count): number_of_uids for count, number_of_uids in sorted(count_histogram.items())
            },
        },
        "errors": errors,
    }
    return validation, uids


def _iter_records(
    new_batch: Any,
    gen_batch: Any,
    uids: Sequence[str],
    tokenizer: Any,
    remove_thinking_section: Callable[..., str | None],
    score_ifeval: Callable[..., float],
    max_prompt_length: int,
    require_think_end: bool,
    progress_every: int,
) -> Iterator[dict[str, Any]]:
    per_uid_index: Counter[str] = Counter()
    size = _batch_size(gen_batch)
    for index in range(size):
        uid = uids[index]
        rollout_index = per_uid_index[uid]
        per_uid_index[uid] += 1

        raw_prompt = _first_non_tensor_value((new_batch, gen_batch), "raw_prompt", index, default=_MISSING)
        if raw_prompt is _MISSING or raw_prompt is None:
            raise KeyError(f"Row {index}: raw_prompt is missing")
        ppl_prompt = _first_non_tensor_value((new_batch, gen_batch), "ppl_prompt", index, default=_MISSING)
        if ppl_prompt is _MISSING or ppl_prompt is None:
            raise KeyError(f"Row {index}: ppl_prompt is missing")
        ground_truth = _extract_ground_truth(new_batch, gen_batch, index)
        extra_info = _first_non_tensor_value((new_batch, gen_batch), "extra_info", index, default={})

        response_token_ids = _extract_response_token_ids(gen_batch, index)
        response_text = _decode(tokenizer, response_token_ids)
        judge_response = remove_thinking_section(response_text, require_think_end=False)
        if judge_response is None:
            judge_response = response_text
        constraint_score = float(
            score_ifeval(response_text, ground_truth, require_think_end=require_think_end)
        )
        ppl_prefix_ids, ppl_prefix_source = _select_ppl_prefix(
            gen_batch,
            tokenizer,
            ppl_prompt,
            max_prompt_length,
            index,
        )
        ppl_continuation_ids, ppl_continuation_source = _select_ppl_continuation(
            gen_batch,
            tokenizer,
            response_token_ids,
            index,
        )
        ppl_eligible = bool(ppl_prefix_ids and ppl_continuation_ids)

        yield {
            "id": f"{uid}:{rollout_index}",
            "row_index": index,
            "uid": uid,
            "rollout_index": rollout_index,
            "raw_prompt": _jsonable(raw_prompt),
            "ppl_prompt": _jsonable(ppl_prompt),
            "ground_truth": _jsonable(ground_truth),
            "extra_info": _jsonable(extra_info),
            "response_token_ids": response_token_ids,
            "response_text": response_text,
            "response_token_count": len(response_token_ids),
            "judge_prompt": _prompt_to_text(ppl_prompt),
            "judge_response": str(judge_response),
            "constraint_score": constraint_score,
            "ppl_eligible": ppl_eligible,
            "ppl_ineligible_reason": "" if ppl_eligible else ppl_continuation_source,
            "ppl_prefix_token_ids": ppl_prefix_ids,
            "ppl_continuation_token_ids": ppl_continuation_ids,
            "ppl_prefix_token_count": len(ppl_prefix_ids),
            "ppl_continuation_token_count": len(ppl_continuation_ids),
            "ppl_prefix_source": ppl_prefix_source,
            "ppl_continuation_source": ppl_continuation_source,
        }

        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"Prepared {index + 1}/{size} rollout rows", file=sys.stderr, flush=True)


def _new_stats() -> dict[str, Any]:
    return {
        "row_count": 0,
        "response_tokens": 0,
        "ppl_prefix_tokens": 0,
        "ppl_continuation_tokens": 0,
        "constraint_score_sum": 0.0,
        "constraint_score_min": None,
        "constraint_score_max": None,
        "ppl_eligible_rows": 0,
        "ppl_prefix_sources": Counter(),
        "ppl_continuation_sources": Counter(),
    }


def _update_stats(stats: dict[str, Any], record: Mapping[str, Any]) -> None:
    stats["row_count"] += 1
    stats["response_tokens"] += int(record["response_token_count"])
    stats["ppl_prefix_tokens"] += int(record["ppl_prefix_token_count"])
    stats["ppl_continuation_tokens"] += int(record["ppl_continuation_token_count"])
    score = float(record["constraint_score"])
    stats["constraint_score_sum"] += score
    stats["ppl_eligible_rows"] += int(bool(record["ppl_eligible"]))
    if stats["constraint_score_min"] is None or score < stats["constraint_score_min"]:
        stats["constraint_score_min"] = score
    if stats["constraint_score_max"] is None or score > stats["constraint_score_max"]:
        stats["constraint_score_max"] = score
    stats["ppl_prefix_sources"][str(record["ppl_prefix_source"])] += 1
    stats["ppl_continuation_sources"][str(record["ppl_continuation_source"])] += 1


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["row_count"])
    return {
        "row_count": count,
        "tokens": {
            "response_total": int(stats["response_tokens"]),
            "response_mean": stats["response_tokens"] / count if count else 0.0,
            "ppl_prefix_total": int(stats["ppl_prefix_tokens"]),
            "ppl_prefix_mean": stats["ppl_prefix_tokens"] / count if count else 0.0,
            "ppl_continuation_total": int(stats["ppl_continuation_tokens"]),
            "ppl_continuation_mean": stats["ppl_continuation_tokens"] / count if count else 0.0,
        },
        "constraint_score": {
            "mean": stats["constraint_score_sum"] / count if count else 0.0,
            "min": stats["constraint_score_min"],
            "max": stats["constraint_score_max"],
        },
        "ppl_eligibility": {
            "eligible_rows": int(stats["ppl_eligible_rows"]),
            "ineligible_rows": count - int(stats["ppl_eligible_rows"]),
        },
        "ppl_prefix_sources": dict(sorted(stats["ppl_prefix_sources"].items())),
        "ppl_continuation_sources": dict(sorted(stats["ppl_continuation_sources"].items())),
    }


def _write_jsonl(records: Iterator[dict[str, Any]], output_path: Path) -> tuple[str, dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    digest = hashlib.sha256()
    stats = _new_stats()
    try:
        with temp_path.open("wb") as handle:
            for record in records:
                normalized = _jsonable(record)
                line = json.dumps(
                    normalized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8") + b"\n"
                handle.write(line)
                digest.update(line)
                _update_stats(stats, normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), _finalize_stats(stats)


def _source_file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _read_source_meta(step_dir: Path) -> Any:
    path = step_dir / "meta.json"
    if not path.is_file():
        return None
    try:
        return _jsonable(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _write_meta(meta_path: Path, meta: Mapping[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = meta_path.with_name(f".{meta_path.name}.tmp-{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(meta), handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, meta_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_dir", type=Path, help="Rollout-skip genstep directory")
    parser.add_argument("--output", type=Path, required=True, help="Canonical compact JSONL path")
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Metadata sidecar path (default: <output>.meta.json)",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=Path(os.getenv("NLTK_DATA", DEFAULT_NLTK_DATA)),
        help="NLTK data directory containing punkt and punkt_tab",
    )
    parser.add_argument(
        "--expected-rows",
        "--expected-responses",
        dest="expected_rows",
        type=int,
        default=DEFAULT_EXPECTED_ROWS,
    )
    parser.add_argument(
        "--expected-unique-uids",
        "--expected-prompts",
        dest="expected_unique_uids",
        type=int,
        default=DEFAULT_EXPECTED_UIDS,
    )
    parser.add_argument(
        "--expected-rollouts-per-uid",
        "--rollouts-per-prompt",
        dest="expected_rollouts_per_uid",
        type=int,
        default=DEFAULT_ROLLOUTS_PER_UID,
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Write the export while recording count/uid validation failures in metadata",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--require-think-end",
        dest="require_think_end",
        action="store_true",
        default=True,
        help="Require </think> for IFEval constraint scoring (default)",
    )
    thinking_group.add_argument(
        "--no-require-think-end",
        dest="require_think_end",
        action="store_false",
        help="Do not require </think> for IFEval constraint scoring",
    )
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    step_dir = args.step_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    meta_path = (
        args.meta_output.expanduser().resolve()
        if args.meta_output is not None
        else Path(str(output_path) + ".meta.json")
    )

    if not step_dir.is_dir():
        raise FileNotFoundError(f"Rollout step directory does not exist: {step_dir}")
    if args.max_prompt_length <= 0:
        raise ValueError("--max-prompt-length must be positive")
    for value, flag in (
        (args.expected_rows, "--expected-rows"),
        (args.expected_unique_uids, "--expected-unique-uids"),
        (args.expected_rollouts_per_uid, "--expected-rollouts-per-uid"),
    ):
        if value <= 0:
            raise ValueError(f"{flag} must be positive")
    if not args.overwrite:
        existing = [str(path) for path in (output_path, meta_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output(s): {', '.join(existing)}")

    print(f"Loading rollout dump: {step_dir}", file=sys.stderr, flush=True)
    new_batch, gen_batch, loader_name = _load_dumped_data(step_dir)
    validation, uids = _rollout_shape(
        new_batch,
        gen_batch,
        expected_rows=args.expected_rows,
        expected_unique_uids=args.expected_unique_uids,
        expected_rollouts_per_uid=args.expected_rollouts_per_uid,
        allow_count_mismatch=args.allow_count_mismatch,
    )
    print(f"Loading tokenizer: {args.tokenizer}", file=sys.stderr, flush=True)
    tokenizer = _load_tokenizer(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    _configure_nltk_data(args.nltk_data)
    remove_thinking_section, score_ifeval = _load_verifier_functions()

    records = _iter_records(
        new_batch,
        gen_batch,
        uids,
        tokenizer,
        remove_thinking_section,
        score_ifeval,
        max_prompt_length=args.max_prompt_length,
        require_think_end=args.require_think_end,
        progress_every=args.progress_every,
    )
    jsonl_sha256, stats = _write_jsonl(records, output_path)
    if stats["row_count"] != validation["observed"]["rows"]:
        raise RuntimeError(
            f"Exporter wrote {stats['row_count']} rows from a {validation['observed']['rows']}-row dump"
        )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "step_dir": str(step_dir),
            "loader": loader_name,
            "new_batch": _source_file_info(step_dir / "new_batch.dp"),
            "gen_batch": _source_file_info(step_dir / "gen_batch.dp"),
            "rollout_meta": _read_source_meta(step_dir),
        },
        "output": {
            "jsonl_path": str(output_path),
            "jsonl_sha256": jsonl_sha256,
            "size_bytes": output_path.stat().st_size,
            "canonical_json": {
                "encoding": "utf-8",
                "sort_keys": True,
                "separators": [",", ":"],
            },
        },
        "tokenizer": {
            "requested": args.tokenizer,
            "revision": args.tokenizer_revision,
            "resolved_name_or_path": str(getattr(tokenizer, "name_or_path", args.tokenizer)),
            "class": type(tokenizer).__name__,
            "max_prompt_length": args.max_prompt_length,
            "ppl_chat_template": {
                "add_generation_prompt": True,
                "enable_thinking": False,
                "left_truncate": True,
            },
        },
        "constraint_scoring": {
            "function": "if_rlvr.ifeval_oi.verifier.score_ifeval",
            "require_think_end": args.require_think_end,
            "nltk_data": str(args.nltk_data.expanduser().resolve()),
        },
        "validation": validation,
        "statistics": stats,
        "canonical_fields": [
            "id",
            "row_index",
            "uid",
            "rollout_index",
            "raw_prompt",
            "ppl_prompt",
            "ground_truth",
            "extra_info",
            "response_token_ids",
            "response_text",
            "response_token_count",
            "judge_prompt",
            "judge_response",
            "constraint_score",
            "ppl_eligible",
            "ppl_ineligible_reason",
            "ppl_prefix_token_ids",
            "ppl_continuation_token_ids",
            "ppl_prefix_token_count",
            "ppl_continuation_token_count",
            "ppl_prefix_source",
            "ppl_continuation_source",
        ],
    }
    _write_meta(meta_path, meta)
    print(
        f"Wrote {stats['row_count']} rows to {output_path}\n"
        f"SHA256 {jsonl_sha256}\n"
        f"Metadata {meta_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
