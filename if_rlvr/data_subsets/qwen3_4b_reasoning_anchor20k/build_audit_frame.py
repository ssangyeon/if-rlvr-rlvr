#!/usr/bin/env python3
"""Build the population audit table used by the Qwen3-4B ~20k curation.

The frame is the *effective* training population: the seed-1 shuffled training
split after the 512-row validation holdout and the launcher's 2,048-token prompt
filter.  On the pinned dataset/tokenizer this is exactly 93,882 of 94,861 rows.

For every eligible input, this script records:

* constraint IDs, families, co-occurrence signature, source, and input lengths;
* prompt lengths under the exact Qwen3 thinking chat template;
* complete/missing status, NLL geometry, flips, output lengths, and hygiene
  flags independently for pinned anchor generations run1, run2, and run3;
* the actual IFEval score of every available constrained anchor response.

Run1's verified 4,096-row cache overrides the inherited full run1 entries.  It
adds the 476 documented 32k gap fills while preserving the original draws for
the other 3,620 nested-subset rows.

The operation is CPU-only and deterministic.  IFEval scoring uses a small,
low-priority process pool; callers should additionally launch this script with
``nice`` while training is live.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HF_ROOT = Path("/data/IFIF/.cache/huggingface/hub")
DATASET_PARQUET = (
    HF_ROOT
    / "datasets--allenai--IF_multi_constraints_upto5"
    / "snapshots/2e3a77407b7fce69f95b248d64a884e3ae1c2423"
    / "data/train-00000-of-00001.parquet"
)
TOKENIZER_PATH = (
    HF_ROOT
    / "models--Qwen--Qwen3-4B"
    / "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
ANCHOR_SNAPSHOTS = HF_ROOT / "datasets--sangyon--anchor_cache" / "snapshots"

RUN_SPECS = {
    "run1": (
        ANCHOR_SNAPSHOTS
        / "e948df687da29e5222dd4cbb37a59e8eeaf3faa7"
        / (
            "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_"
            "r8192_scored_by_qwen3_4b.json"
        ),
        "57a8128860896145cf22607b3d6e33327b3735f77a9ac239e6e05fbf6885937f",
    ),
    "run2": (
        ANCHOR_SNAPSHOTS
        / "bc72af3622590af3459181932e3e4949c162c0e8"
        / (
            "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_"
            "r8192_plus32768fallback_run2_scored_by_qwen3_4b.json"
        ),
        "5c9da5d1b4ce0fb6b981f6dbdcaf53b51238a227f593d3e767a32b4d8dd6e765",
    ),
    "run3": (
        ANCHOR_SNAPSHOTS
        / "0e030ca1600da5306e5474985137060b7231d254"
        / (
            "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_"
            "r8192_plus32768fallback_run3_scored_by_qwen3_4b.json"
        ),
        "57a6e296964f3c4849fdc3543a8c665e6be62052ef093e6d70b942e3971f9cec",
    ),
}
RUN1_SUBSET_PATH = (
    ANCHOR_SNAPSHOTS
    / "e948df687da29e5222dd4cbb37a59e8eeaf3faa7"
    / (
        "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_"
        "r8192_scored_by_qwen3_4b.SUBSET4096.json"
    )
)
RUN1_SUBSET_SHA256 = "5ba0ba126d3677e816334e5c4e43cccfe019c3f68d61fe420f16067eebf5f0a8"

EXPECTED_METADATA = {
    "version": 3,
    "model_path": "Qwen/Qwen3-4B",
    "if_dataset_seed": 1,
    "if_dataset_val_size": 512,
    "max_prompt_length": 2048,
    "max_response_length": 8192,
    "rollout_temperature": 1.0,
    "rollout_top_p": 0.95,
    "rollout_top_k": 20,
    "ppl_prefix_mode": "standard",
    "ppl_nll_scope": "final_answer_tokens_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_constraint_spec(label: str) -> tuple[list[str], list[Any]]:
    value = ast.literal_eval(label)[0]
    if isinstance(value, str):
        value = json.loads(value)
    return list(value["instruction_id"]), list(value["kwargs"])


def constraint_free_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Mirror if_dataset._make_constraint_free_messages without importing verl."""
    parts = [part.strip() for part in (example.get("constraint") or "").split("\t") if part.strip()]
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    for message in messages:
        if message["role"] != "user" or not parts:
            continue
        content = message.get("content", "")
        found = [content.find(part) for part in parts]
        found = [position for position in found if position >= 0]
        if not found:
            continue
        base = content[: min(found)].strip()
        if not base or any(part in base for part in parts):
            base = content
            for part in sorted(parts, key=len, reverse=True):
                base = base.replace(part, " ")
            base = re.sub(r"[ \t]{2,}", " ", base).strip()
        message["content"] = base
    return messages


def script_ratios(text: str) -> dict[str, float]:
    denom = max(len(text), 1)
    counts = Counter()
    for char in text:
        code = ord(char)
        if code < 128:
            counts["ascii"] += 1
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["cjk"] += 1
        elif 0x0400 <= code <= 0x052F:
            counts["cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["arabic"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            counts["hangul"] += 1
        elif code >= 128 and unicodedata.category(char).startswith("L"):
            counts["other_nonascii_letter"] += 1
    return {f"x_{name}_ratio": counts[name] / denom for name in (
        "ascii", "cjk", "cyrillic", "arabic", "devanagari", "hangul", "other_nonascii_letter"
    )}


def complete_item(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    try:
        count0 = int(item["ref0_token_count"])
        count1 = int(item["ref1_token_count"])
        nll0 = float(item["ref0_nll"])
        nll1 = float(item["ref1_nll"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        item.get("y0")
        and item.get("y1")
        and count0 > 0
        and count1 > 0
        and math.isfinite(nll0)
        and math.isfinite(nll1)
    )


def is_loop_text(text: str) -> bool:
    words = text.split()
    count = len(words)
    if count < 50:
        return False
    if len(set(words)) / count < 0.15:
        return True
    grams = Counter(zip(*(words[offset:] for offset in range(8))))
    return bool(grams and grams.most_common(1)[0][1] / max(count - 7, 1) > 0.5)


def _score_chunk(records: list[tuple[str, str]]) -> list[float]:
    # Spawned workers import only the lightweight self-contained verifier.
    os.environ.setdefault("NLTK_DATA", "/data/IFIF/IFBench/.nltk_data")
    if str(ROOT / "if_rlvr") not in sys.path:
        sys.path.insert(0, str(ROOT / "if_rlvr"))
    logging.getLogger("ifeval_oi.verifier").setLevel(logging.ERROR)
    try:
        from langdetect import DetectorFactory

        DetectorFactory.seed = 0
    except Exception:
        pass
    from ifeval_oi.verifier import score_ifeval

    return [float(score_ifeval(text, label, require_think_end=False)) for text, label in records]


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def validate_metadata(payload: dict[str, Any], run: str) -> None:
    metadata = payload.get("metadata", {})
    bad = {
        key: (metadata.get(key), expected)
        for key, expected in EXPECTED_METADATA.items()
        if metadata.get(key) != expected
    }
    thinking = metadata.get("apply_chat_template_kwargs", {}).get("enable_thinking")
    if thinking is not True:
        bad["apply_chat_template_kwargs.enable_thinking"] = (thinking, True)
    if bad:
        raise RuntimeError(f"{run}: incompatible metadata: {bad}")


def anchor_columns(size: int, run: str) -> dict[str, np.ndarray]:
    return {
        f"{run}_available": np.zeros(size, dtype=bool),
        f"{run}_m0": np.full(size, np.nan),
        f"{run}_m1": np.full(size, np.nan),
        f"{run}_width": np.full(size, np.nan),
        f"{run}_y0_tokens": np.full(size, -1, dtype=np.int32),
        f"{run}_y1_tokens": np.full(size, -1, dtype=np.int32),
        f"{run}_flipped": np.zeros(size, dtype=bool),
        f"{run}_ifscore": np.full(size, np.nan),
        f"{run}_allsat": np.zeros(size, dtype=bool),
        f"{run}_short_y0": np.zeros(size, dtype=bool),
        f"{run}_short_y1": np.zeros(size, dtype=bool),
        f"{run}_loop_y0": np.zeros(size, dtype=bool),
        f"{run}_loop_y1": np.zeros(size, dtype=bool),
        f"{run}_duplicate": np.zeros(size, dtype=bool),
        f"{run}_cot_marker": np.zeros(size, dtype=bool),
    }


def load_anchor_run(
    run: str,
    path: Path,
    expected_sha256: str,
    labels: list[str],
    tokenizer: Any,
    pool: Any,
    size: int,
    *,
    run1_subset_payload: dict[str, Any] | None = None,
    decode_batch: int = 1024,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.time()
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(f"{run}: SHA256 {actual_sha} != pinned {expected_sha256}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    validate_metadata(payload, run)
    items = payload["items"]
    source_items = len(items)
    overrides = 0
    if run1_subset_payload is not None:
        validate_metadata(run1_subset_payload, "run1_subset")
        overrides = len(run1_subset_payload["items"])
        items.update(run1_subset_payload["items"])

    columns = anchor_columns(size, run)
    valid_keys: list[int] = []
    token_mismatches = 0
    ppl_mismatches = 0
    for raw_key, item in items.items():
        index = int(raw_key)
        if index < 0 or index >= size or not complete_item(item):
            continue
        count0 = int(item["ref0_token_count"])
        count1 = int(item["ref1_token_count"])
        nll0 = float(item["ref0_nll"])
        nll1 = float(item["ref1_nll"])
        m0, m1 = nll0 / count0, nll1 / count1
        if count0 != len(item["y0"]) or count1 != len(item["y1"]):
            token_mismatches += 1
            continue
        if not math.isclose(float(item.get("ref0_ppl", math.exp(m0))), math.exp(m0), rel_tol=1e-6):
            ppl_mismatches += 1
        if not math.isclose(float(item.get("ref1_ppl", math.exp(m1))), math.exp(m1), rel_tol=1e-6):
            ppl_mismatches += 1
        columns[f"{run}_available"][index] = True
        columns[f"{run}_m0"][index] = m0
        columns[f"{run}_m1"][index] = m1
        columns[f"{run}_width"][index] = m1 - m0
        columns[f"{run}_y0_tokens"][index] = count0
        columns[f"{run}_y1_tokens"][index] = count1
        columns[f"{run}_flipped"][index] = m1 < m0
        columns[f"{run}_short_y0"][index] = count0 < 10
        columns[f"{run}_short_y1"][index] = count1 < 10
        columns[f"{run}_duplicate"][index] = item["y0"] == item["y1"]
        valid_keys.append(index)

    # Decode and verify in bounded batches; the decoded strings are never retained.
    scores_written = 0
    for key_batch in chunks(valid_keys, decode_batch):
        y1_ids = [items[str(index)]["y1"] for index in key_batch]
        y0_ids = [items[str(index)]["y0"] for index in key_batch]
        y1_text = tokenizer.batch_decode(y1_ids, skip_special_tokens=True)
        y0_text = tokenizer.batch_decode(y0_ids, skip_special_tokens=True)
        records = list(zip(y1_text, (labels[index] for index in key_batch), strict=True))
        work = list(chunks(records, 64))
        scored_parts = pool.map(_score_chunk, work)
        scores = [score for part in scored_parts for score in part]
        for offset, index in enumerate(key_batch):
            score = scores[offset]
            columns[f"{run}_ifscore"][index] = score
            columns[f"{run}_allsat"][index] = math.isclose(score, 1.0)
            columns[f"{run}_loop_y0"][index] = is_loop_text(y0_text[offset])
            columns[f"{run}_loop_y1"][index] = is_loop_text(y1_text[offset])
            columns[f"{run}_cot_marker"][index] = any(
                marker in y1_text[offset] for marker in ("<think>", "</think>", "<|think|>")
            )
        scores_written += len(key_batch)
        if scores_written % (decode_batch * 10) == 0:
            print(f"  {run}: decoded/scored {scores_written:,}/{len(valid_keys):,}", flush=True)

    report = {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "file_items": source_items,
        "run1_subset_overrides": overrides,
        "complete_rows_after_override": len(valid_keys),
        "token_count_mismatches_rejected": token_mismatches,
        "ppl_consistency_mismatches": ppl_mismatches,
        "seconds": time.time() - started,
    }
    del payload, items
    gc.collect()
    return columns, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--decode-batch", type=int, default=1024)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NLTK_DATA", "/data/IFIF/IFBench/.nltk_data")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    started = time.time()
    dataset = load_dataset("parquet", data_files=str(DATASET_PARQUET), split="train")
    raw_rows = len(dataset)
    dataset = dataset.shuffle(seed=1).select(range(512, raw_rows))
    train_rows = len(dataset)
    labels = list(dataset["ground_truth"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)

    input_rows: list[dict[str, Any]] = []
    full_conversations: list[list[dict[str, str]]] = []
    x_conversations: list[list[dict[str, str]]] = []
    for index in range(train_rows):
        example = dataset[index]
        instruction_ids, _ = parse_constraint_spec(example["ground_truth"])
        families = [value.split(":", 1)[0] for value in instruction_ids]
        x_messages = constraint_free_messages(example)
        x_text = "\n".join(m["content"] for m in x_messages if m["role"] == "user")
        prompt_text = "\n".join(m["content"] for m in example["messages"] if m["role"] == "user")
        row = {
            "index": index,
            "dataset_key": example.get("key", ""),
            "source_group": (example.get("key", "").split("/", 1)[0] or "unknown"),
            "constraint_type": example.get("constraint_type", ""),
            "constraint_ids_json": json.dumps(instruction_ids, separators=(",", ":")),
            "constraint_families_json": json.dumps(families, separators=(",", ":")),
            "constraint_signature": "|".join(sorted(instruction_ids)),
            "n_constraints": len(instruction_ids),
            "x_chars": len(x_text),
            "constraint_chars": len(example.get("constraint") or ""),
            "prompt_chars": len(prompt_text),
            "x_words": len(x_text.split()),
            "prompt_words": len(prompt_text.split()),
            "x_lines": x_text.count("\n") + 1,
            **script_ratios(x_text),
        }
        input_rows.append(row)
        full_conversations.append(example["messages"])
        x_conversations.append(x_messages)

    prompt_tokens = np.empty(train_rows, dtype=np.int32)
    x_prompt_tokens = np.empty(train_rows, dtype=np.int32)
    for start in range(0, train_rows, 512):
        stop = min(start + 512, train_rows)
        full_ids = tokenizer.apply_chat_template(
            full_conversations[start:stop],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        x_ids = tokenizer.apply_chat_template(
            x_conversations[start:stop],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_tokens[start:stop] = [len(value) for value in full_ids]
        x_prompt_tokens[start:stop] = [len(value) for value in x_ids]
    del full_conversations, x_conversations

    frame = pd.DataFrame(input_rows)
    frame["prompt_tokens"] = prompt_tokens
    frame["x_prompt_tokens"] = x_prompt_tokens
    frame["constraint_tokens"] = prompt_tokens - x_prompt_tokens
    eligible = prompt_tokens <= 2048

    subset_sha = sha256_file(RUN1_SUBSET_PATH)
    if subset_sha != RUN1_SUBSET_SHA256:
        raise RuntimeError(f"run1 subset SHA256 {subset_sha} != pinned {RUN1_SUBSET_SHA256}")
    with RUN1_SUBSET_PATH.open("r", encoding="utf-8") as stream:
        run1_subset_payload = json.load(stream)

    ctx = mp.get_context("spawn")
    run_reports: dict[str, Any] = {}
    with ctx.Pool(processes=args.workers) as pool:
        for run, (path, expected_sha) in RUN_SPECS.items():
            print(f"loading and auditing {run}: {path}", flush=True)
            columns, report = load_anchor_run(
                run,
                path,
                expected_sha,
                labels,
                tokenizer,
                pool,
                train_rows,
                run1_subset_payload=run1_subset_payload if run == "run1" else None,
                decode_batch=args.decode_batch,
            )
            for name, values in columns.items():
                frame[name] = values
            run_reports[run] = report
            print(f"finished {run}: {report}", flush=True)

    # The anchor cache key set and prompt eligibility must agree exactly.  This
    # proves that the curation frame matches what the configured trainer sees.
    with RUN_SPECS["run1"][0].open("r", encoding="utf-8") as stream:
        run1_file_keys = {int(key) for key in json.load(stream)["items"]}
    eligible_keys = set(frame.loc[eligible, "index"].astype(int))
    if run1_file_keys != eligible_keys:
        raise RuntimeError(
            f"prompt-filter/cache-key mismatch: eligible_only={len(eligible_keys-run1_file_keys)} "
            f"cache_only={len(run1_file_keys-eligible_keys)}"
        )

    frame = frame.loc[eligible].reset_index(drop=True)
    if len(frame) != 93_882:
        raise RuntimeError(f"expected 93,882 eligible rows, found {len(frame)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, compression="zstd")

    provenance = {
        "version": 1,
        "created_unix": time.time(),
        "dataset_parquet": str(DATASET_PARQUET.resolve()),
        "dataset_snapshot": "2e3a77407b7fce69f95b248d64a884e3ae1c2423",
        "tokenizer_path": str(TOKENIZER_PATH.resolve()),
        "split": {"raw_rows": raw_rows, "validation_rows": 512, "train_rows": train_rows},
        "prompt_filter": {
            "max_prompt_tokens": 2048,
            "eligible_rows": int(eligible.sum()),
            "excluded_rows": int((~eligible).sum()),
            "exactly_matches_run1_cache_keyset": True,
        },
        "run1_subset_override": {
            "path": str(RUN1_SUBSET_PATH.resolve()),
            "sha256": subset_sha,
        },
        "anchor_runs": run_reports,
        "ifeval_scoring": {
            "workers": args.workers,
            "require_think_end": False,
            "nltk_data": os.environ["NLTK_DATA"],
            "langdetect_seed": 0,
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "seconds": time.time() - started,
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
