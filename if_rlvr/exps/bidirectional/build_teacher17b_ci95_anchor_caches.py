#!/usr/bin/env python3
"""Build seed-aligned 95% CI anchor caches from teacher17b cache files.

The cache item index is local to the dataset shuffle seed, so entries from
different seeds cannot be joined by their JSON key.  This script reconstructs
the original Hugging Face dataset row index for every seed and joins on that
stable index before computing the confidence intervals.

The GRPO reward compares per-token mean NLL values:

    lower threshold = ref0_nll / ref0_token_count
    upper threshold = ref1_nll / ref1_token_count

For each threshold, this script computes a two-sided Student t interval over
the eight seed values.  The output cache variants are:

* outer: lower CI lower endpoint + upper CI upper endpoint
* inner: lower CI upper endpoint + upper CI lower endpoint

Response token IDs and token counts are copied from the seed-1 item.  NLL and
PPL fields are rewritten consistently with the selected CI endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets


DEFAULT_CACHE_ROOT = Path(
    "/NHNHOME/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr/.cache"
)
DEFAULT_SOURCE_PATTERN = (
    "if_ref_anchor_teacher17b_nonreason_train_seed{seed}_val512_scored_by_qwen3_17b.json"
)
DEFAULT_OUTER_NAME = (
    "if_ref_anchor_teacher17b_nonreason_train_seed1_val512_scored_by_qwen3_17b_"
    "ci95_outer_s1to8.json"
)
DEFAULT_INNER_NAME = (
    "if_ref_anchor_teacher17b_nonreason_train_seed1_val512_scored_by_qwen3_17b_"
    "ci95_inner_s1to8.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--source-pattern", default=DEFAULT_SOURCE_PATTERN)
    parser.add_argument("--dataset", default="allenai/IF_multi_constraints_upto5")
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--output-seed", type=int, default=1)
    parser.add_argument("--outer-output", type=Path)
    parser.add_argument("--inner-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def t_critical_975(df: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=df))
    except ImportError:
        if df == 7:
            return 2.3646242510102993
        raise RuntimeError("scipy is required when the CI does not use df=7") from None


def interval(values: list[float], t_critical: float) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("At least two seed values are required for a confidence interval")
    mean = statistics.fmean(values)
    half_width = t_critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width, mean


def atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, separators=(",", ":"), allow_nan=False)
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def complete_thresholds(item: dict[str, Any]) -> tuple[float, float] | None:
    try:
        lower_nll = float(item["ref0_nll"])
        lower_count = int(item["ref0_token_count"])
        upper_nll = float(item["ref1_nll"])
        upper_count = int(item["ref1_token_count"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not item.get("y0")
        or not item.get("y1")
        or lower_count <= 0
        or upper_count <= 0
        or not math.isfinite(lower_nll)
        or not math.isfinite(upper_nll)
    ):
        return None
    return lower_nll / lower_count, upper_nll / upper_count


def rewrite_item(item: dict[str, Any], lower_mean_nll: float, upper_mean_nll: float) -> None:
    lower_count = int(item["ref0_token_count"])
    upper_count = int(item["ref1_token_count"])
    item["ref0_nll"] = float(lower_mean_nll * lower_count)
    item["ref0_ppl"] = float(math.exp(lower_mean_nll))
    item["ref1_nll"] = float(upper_mean_nll * upper_count)
    item["ref1_ppl"] = float(math.exp(upper_mean_nll))


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("Provide at least two distinct seeds")
    if args.output_seed not in seeds:
        raise ValueError("--output-seed must be included in --seeds")

    outer_output = args.outer_output or args.cache_root / DEFAULT_OUTER_NAME
    inner_output = args.inner_output or args.cache_root / DEFAULT_INNER_NAME
    for output_path in (outer_output, inner_output):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists (use --overwrite): {output_path}")

    print(f"Loading dataset mapping: {args.dataset}", flush=True)
    dataset = datasets.load_dataset(args.dataset, split="train")
    original_index_column = "__ci95_original_index"
    if original_index_column in dataset.column_names:
        raise RuntimeError(f"Temporary column already exists: {original_index_column}")
    dataset = dataset.add_column(original_index_column, list(range(len(dataset))))

    seed_to_original_indices: dict[int, list[int]] = {}
    for seed in seeds:
        shuffled = dataset.shuffle(seed=seed)
        seed_to_original_indices[seed] = list(shuffled[original_index_column][args.val_size :])
        print(
            f"seed={seed}: reconstructed {len(seed_to_original_indices[seed])} train indices",
            flush=True,
        )

    lower_values: dict[int, list[float]] = defaultdict(list)
    upper_values: dict[int, list[float]] = defaultdict(list)
    output_payload: dict[str, Any] | None = None
    output_original_to_key: dict[int, str] = {}
    source_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}

    for seed in seeds:
        source_path = args.cache_root / args.source_pattern.format(seed=seed)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source cache: {source_path}")
        print(f"Loading seed={seed}: {source_path}", flush=True)
        with source_path.open("r", encoding="utf-8") as source_file:
            payload = json.load(source_file)

        metadata_seed = payload.get("metadata", {}).get("if_dataset_seed")
        if int(metadata_seed) != seed:
            raise ValueError(
                f"Seed metadata mismatch in {source_path}: expected {seed}, got {metadata_seed}"
            )

        items = payload.get("items", {})
        original_indices = seed_to_original_indices[seed]
        seen_original_indices: set[int] = set()
        valid_count = 0
        invalid_count = 0
        for key, item in items.items():
            try:
                local_index = int(key)
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            if local_index < 0 or local_index >= len(original_indices):
                raise IndexError(
                    f"Cache index {local_index} is outside seed={seed} train split "
                    f"of length {len(original_indices)}"
                )
            original_index = int(original_indices[local_index])
            if original_index in seen_original_indices:
                raise RuntimeError(f"Duplicate original index for seed={seed}: {original_index}")
            seen_original_indices.add(original_index)

            thresholds = complete_thresholds(item)
            if thresholds is None:
                invalid_count += 1
                continue
            lower, upper = thresholds
            lower_values[original_index].append(lower)
            upper_values[original_index].append(upper)
            valid_count += 1

            if seed == args.output_seed:
                output_original_to_key[original_index] = str(key)

        source_counts[str(seed)] = valid_count
        invalid_counts[str(seed)] = invalid_count
        print(
            f"seed={seed}: valid={valid_count}, invalid={invalid_count}, file_items={len(items)}",
            flush=True,
        )

        if seed == args.output_seed:
            output_payload = payload

    if output_payload is None:
        raise RuntimeError("The output-seed payload was not loaded")

    sample_count = len(seeds)
    common_original_indices = [
        original_index
        for original_index in output_original_to_key
        if len(lower_values[original_index]) == sample_count
        and len(upper_values[original_index]) == sample_count
    ]
    common_original_indices.sort(key=lambda value: int(output_original_to_key[value]))
    if not common_original_indices:
        raise RuntimeError("No samples are present in every seed cache")

    critical = t_critical_975(sample_count - 1)
    ci_by_original: dict[int, tuple[float, float, float, float]] = {}
    outer_crossed = 0
    inner_crossed = 0
    negative_outer_lower = 0
    for original_index in common_original_indices:
        lower_low, lower_high, _ = interval(lower_values[original_index], critical)
        upper_low, upper_high, _ = interval(upper_values[original_index], critical)
        ci_by_original[original_index] = (lower_low, lower_high, upper_low, upper_high)
        outer_crossed += int(lower_low > upper_high)
        inner_crossed += int(lower_high > upper_low)
        negative_outer_lower += int(lower_low < 0.0)

    output_items = output_payload["items"]
    selected_items = {
        output_original_to_key[original_index]: output_items[output_original_to_key[original_index]]
        for original_index in common_original_indices
    }
    output_payload["items"] = selected_items

    common_ci_metadata: dict[str, Any] = {
        "version": 1,
        "confidence_level": 0.95,
        "method": "two-sided Student t interval on per-token mean NLL",
        "sample_count": sample_count,
        "degrees_of_freedom": sample_count - 1,
        "t_critical_0.975": critical,
        "seeds": seeds,
        "output_dataset_seed": args.output_seed,
        "dataset": args.dataset,
        "validation_size": args.val_size,
        "alignment": (
            "original HF dataset row index reconstructed before per-seed shuffle and val holdout"
        ),
        "lower_threshold": "ref0_nll / ref0_token_count",
        "upper_threshold": "ref1_nll / ref1_token_count",
        "source_pattern": args.source_pattern,
        "source_valid_item_counts": source_counts,
        "source_invalid_item_counts": invalid_counts,
        "common_item_count": len(common_original_indices),
        "outer_lower_below_zero_count": negative_outer_lower,
        "outer_lower_above_upper_count": outer_crossed,
        "inner_lower_above_upper_count": inner_crossed,
        "payload_fields_copied_from_seed": args.output_seed,
    }

    print(
        "CI summary: "
        f"common={len(common_original_indices)}, "
        f"outer_crossed={outer_crossed}, inner_crossed={inner_crossed}, "
        f"outer_lower_below_zero={negative_outer_lower}, t_critical={critical:.12f}",
        flush=True,
    )

    for original_index in common_original_indices:
        lower_low, _, _, upper_high = ci_by_original[original_index]
        rewrite_item(selected_items[output_original_to_key[original_index]], lower_low, upper_high)
    output_payload["ci95"] = {
        **common_ci_metadata,
        "variant": "outer",
        "selected_endpoints": {
            "lower_threshold": "lower CI endpoint",
            "upper_threshold": "upper CI endpoint",
        },
    }
    print(f"Writing outer cache: {outer_output}", flush=True)
    atomic_json_dump(output_payload, outer_output)

    for original_index in common_original_indices:
        _, lower_high, upper_low, _ = ci_by_original[original_index]
        rewrite_item(selected_items[output_original_to_key[original_index]], lower_high, upper_low)
    output_payload["ci95"] = {
        **common_ci_metadata,
        "variant": "inner",
        "selected_endpoints": {
            "lower_threshold": "upper CI endpoint",
            "upper_threshold": "lower CI endpoint",
        },
    }
    print(f"Writing inner cache: {inner_output}", flush=True)
    atomic_json_dump(output_payload, inner_output)

    print("Done", flush=True)
    print(f"outer={outer_output}", flush=True)
    print(f"inner={inner_output}", flush=True)


if __name__ == "__main__":
    main()
