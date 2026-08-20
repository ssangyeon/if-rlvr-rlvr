#!/usr/bin/env python3
"""Materialize and validate the representative Qwen3-4B 20K data panel.

The curation manifest is the authority for input selection.  This utility:

* reconstructs the exact seed-1 / validation-512 shuffled IF training split;
* writes the selected raw rows with their stable post-split training indices;
* carves every currently complete selected item from pinned anchor runs 1--3;
* writes exact missing-index manifests and four cost-balanced *data* shards;
* can validate and merge future 32K fallback generations without changing the
  already-selected training population.

It never treats an ``AVAILABLE`` cache as a complete 20,480-row training cache.
Final caches are emitted only after every selected key has a complete item.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNTIME_ROOT_DEFAULT = ROOT / ".agent_runtime" / "subset20k"
MANIFEST_DEFAULT = HERE / "subset_indices.json"
AUDIT_DEFAULT = ROOT / ".agent_runtime" / "subset20k" / "selected_audit.parquet"

DATASET_PARQUET = (
    Path("/data/IFIF/.cache/huggingface/hub")
    / "datasets--allenai--IF_multi_constraints_upto5"
    / "snapshots/2e3a77407b7fce69f95b248d64a884e3ae1c2423"
    / "data/train-00000-of-00001.parquet"
)
ANCHOR_ROOT = Path("/data/IFIF/.cache/huggingface/hub") / "datasets--sangyon--anchor_cache" / "snapshots"
RUN_SPECS = {
    "run1": {
        "path": ANCHOR_ROOT
        / "e948df687da29e5222dd4cbb37a59e8eeaf3faa7"
        / ("if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.json"),
        "sha256": "57a8128860896145cf22607b3d6e33327b3735f77a9ac239e6e05fbf6885937f",
        "revision": "e948df687da29e5222dd4cbb37a59e8eeaf3faa7",
    },
    "run2": {
        "path": ANCHOR_ROOT
        / "bc72af3622590af3459181932e3e4949c162c0e8"
        / (
            "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_"
            "r8192_plus32768fallback_run2_scored_by_qwen3_4b.json"
        ),
        "sha256": "5c9da5d1b4ce0fb6b981f6dbdcaf53b51238a227f593d3e767a32b4d8dd6e765",
        "revision": "bc72af3622590af3459181932e3e4949c162c0e8",
    },
    "run3": {
        "path": ANCHOR_ROOT
        / "0e030ca1600da5306e5474985137060b7231d254"
        / (
            "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_"
            "r8192_plus32768fallback_run3_scored_by_qwen3_4b.json"
        ),
        "sha256": "57a6e296964f3c4849fdc3543a8c665e6be62052ef093e6d70b942e3971f9cec",
        "revision": "0e030ca1600da5306e5474985137060b7231d254",
    },
}
RUN1_OVERRIDE = {
    "path": ANCHOR_ROOT
    / "e948df687da29e5222dd4cbb37a59e8eeaf3faa7"
    / ("if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json"),
    "sha256": "5ba0ba126d3677e816334e5c4e43cccfe019c3f68d61fe420f16067eebf5f0a8",
}

EXPECTED_SOURCE_METADATA = {
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
EXPECTED_GENERATION_METADATA = {
    **EXPECTED_SOURCE_METADATA,
    "max_response_length": 32768,
}
N_SELECTED = 20_480
N_FIXED = 4_096
N_SHARDS = 4
EXPECTED_MISSING_ROWS = {"run1": 2_664, "run2": 1_059, "run3": 1_066}
EXPECTED_MISSING_UNION = 3_034
EXPECTED_MISSING_DRAWS = 4_789
EXPECTED_GENERATED_COMPLETIONS = 2 * EXPECTED_MISSING_DRAWS
GENERATION_AGENT_WORKERS = {"run1": 72, "run2": 106, "run3": 82}
GENERATION_BATCH_ROWS = {"run1": 288, "run2": 318, "run3": 328}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keyset_sha256(keys: set[str] | list[str]) -> str:
    ordered = sorted((str(int(key)) for key in keys), key=int)
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


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
        item.get("y0") and item.get("y1") and count0 > 0 and count1 > 0 and math.isfinite(nll0) and math.isfinite(nll1)
    )


def generated_item_train_ready(item: dict[str, Any] | None) -> bool:
    if not complete_item(item):
        return False
    assert item is not None
    try:
        y0 = item["y0"]
        y1 = item["y1"]
        count0 = int(item["ref0_token_count"])
        count1 = int(item["ref1_token_count"])
        ppl0 = float(item["ref0_ppl"])
        ppl1 = float(item["ref1_ppl"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        isinstance(y0, list)
        and isinstance(y1, list)
        and count0 == len(y0)
        and count1 == len(y1)
        and len(y0) <= 32768
        and len(y1) <= 32768
        and math.isfinite(ppl0)
        and math.isfinite(ppl1)
    )


def atomic_json(payload: Any, destination: Path, *, pretty: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2 if pretty else None,
                sort_keys=pretty,
                separators=None if pretty else (",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def validate_metadata(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    metadata = payload.get("metadata", {})
    bad = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    thinking = metadata.get("apply_chat_template_kwargs", {}).get("enable_thinking")
    if thinking is not True:
        bad["apply_chat_template_kwargs.enable_thinking"] = (thinking, True)
    if bad:
        raise RuntimeError(f"{label}: metadata mismatch: {bad}")


def load_manifest(path: Path) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    manifest = load_json(path)
    rows = manifest.get("rows", [])
    keys = [str(int(row["index"])) for row in rows]
    panels = {str(int(row["index"])): str(row["panel"]) for row in rows}
    if len(keys) != N_SELECTED or len(set(keys)) != N_SELECTED:
        raise RuntimeError(f"selection must contain {N_SELECTED} unique indices")
    counts = {name: sum(value == name for value in panels.values()) for name in "ABCDE"}
    if counts != {name: N_FIXED for name in "ABCDE"}:
        raise RuntimeError(f"panel counts are invalid: {counts}")
    return manifest, sorted(keys, key=int), panels


def load_sources(selected: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    available: dict[str, dict[str, Any]] = {}
    for run, spec in RUN_SPECS.items():
        path = Path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"{run}: source SHA256 {actual} != {spec['sha256']}")
        payload = load_json(path)
        if not isinstance(payload.get("items"), dict):
            raise RuntimeError(f"{run}: source has no items mapping")
        validate_metadata(payload, EXPECTED_SOURCE_METADATA, run)
        payloads[run] = payload

    override_path = Path(RUN1_OVERRIDE["path"])
    actual = sha256_file(override_path)
    if actual != RUN1_OVERRIDE["sha256"]:
        raise RuntimeError(f"run1 override SHA256 {actual} != {RUN1_OVERRIDE['sha256']}")
    override = load_json(override_path)
    validate_metadata(override, EXPECTED_SOURCE_METADATA, "run1 override")
    override_items = override.get("items", {})
    if len(override_items) != N_FIXED or any(not complete_item(item) for item in override_items.values()):
        raise RuntimeError("run1 override is not the verified complete 4,096-row cache")

    selected_set = set(selected)
    if not set(override_items).issubset(selected_set):
        raise RuntimeError("the selected set no longer contains every fixed 4K override key")
    for run in RUN_SPECS:
        source_items = payloads[run]["items"]
        current: dict[str, Any] = {}
        for key in selected:
            item = override_items.get(key, source_items.get(key)) if run == "run1" else source_items.get(key)
            if complete_item(item):
                current[key] = item
        available[run] = current
    return payloads, available


def observed_cost(key: str, available: dict[str, dict[str, Any]]) -> tuple[int, bool]:
    costs: list[int] = []
    for items in available.values():
        item = items.get(key)
        if item is not None:
            costs.append(int(item["ref0_token_count"]) + int(item["ref1_token_count"]))
    # A row missing from all three draws is empirically the hard tail. Assign
    # the full two-response budget so these rows are spread evenly, rather than
    # pretending their generation cost is negligible.
    return (max(costs), False) if costs else (2 * 32768, True)


def make_shards(
    run: str,
    missing: set[str],
    available: dict[str, dict[str, Any]],
    runtime_root: Path,
) -> list[dict[str, Any]]:
    bins: list[list[tuple[str, int, bool]]] = [[] for _ in range(N_SHARDS)]
    totals = [0] * N_SHARDS
    weighted = []
    for key in missing:
        weight, no_evidence = observed_cost(key, available)
        weighted.append((key, weight, no_evidence))
    weighted.sort(key=lambda value: (-value[1], int(value[0])))
    for entry in weighted:
        shard = min(range(N_SHARDS), key=lambda value: (totals[value], len(bins[value]), value))
        bins[shard].append(entry)
        totals[shard] += entry[1]

    manifests = []
    manifest_root = HERE / "generation_manifests" / run
    output_root = runtime_root / "generated_shards" / run
    for shard, entries in enumerate(bins):
        keys = sorted((value[0] for value in entries), key=int)
        payload = {
            "version": 1,
            "run": run,
            "shard": shard,
            "indices": [int(key) for key in keys],
            "row_count": len(keys),
            "estimated_token_cost": sum(value[1] for value in entries),
            "no_observed_draw_cost_assigned_full_budget": sum(value[2] for value in entries),
            "keyset_sha256": keyset_sha256(set(keys)),
            "selected_keyset_sha256": keyset_sha256(set(available[run]) | missing),
            "output_cache": str((output_root / f"shard{shard}.cache.json").resolve()),
            "generation": {
                "model": "Qwen/Qwen3-4B",
                "enable_thinking": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0.0,
                "max_prompt_tokens": 2048,
                "max_response_tokens": 32768,
                "tensor_model_parallel_size": 1,
                "data_shards": 4,
                "sample_seed": {"run1": 11000, "run2": 22000, "run3": 33000}[run] + shard,
            },
        }
        atomic_json(payload, manifest_root / f"shard{shard}.indices.json", pretty=True)
        manifests.append(payload)
    if set().union(*(set(map(str, item["indices"])) for item in manifests)) != missing:
        raise RuntimeError(f"{run}: shard union does not equal the missing set")
    if sum(item["row_count"] for item in manifests) != len(missing):
        raise RuntimeError(f"{run}: shard row counts do not sum to the missing set")
    return manifests


def write_selected_dataset(selected: list[str], panels: dict[str, str], destination: Path) -> dict[str, Any]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(DATASET_PARQUET), split="train")
    raw_rows = len(dataset)
    train = dataset.shuffle(seed=1).select(range(512, raw_rows))
    positions = [int(key) for key in selected]
    subset = train.select(positions)
    subset = subset.add_column("if_train_index", positions)
    subset = subset.add_column("subset20k_panel", [panels[key] for key in selected])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    subset.to_parquet(str(temporary))
    os.replace(temporary, destination)
    if list(subset["if_train_index"]) != positions:
        raise RuntimeError("selected parquet row order changed")
    return {
        "raw_rows": raw_rows,
        "validation_rows": 512,
        "post_split_train_rows": len(train),
        "selected_rows": len(subset),
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
    }


def available_cache_path(runtime_root: Path, run: str) -> Path:
    return runtime_root / "anchor_views" / f"{run}.SUBSET20480.AVAILABLE.json"


def complete_cache_path(runtime_root: Path, run: str) -> Path:
    return runtime_root / "anchor_views" / f"{run}.SUBSET20480.complete.json"


def generated_run_cache_path(runtime_root: Path, run: str) -> Path:
    """One-shot output from a four-replica, tensor-parallel-size-one run."""
    return runtime_root / "generated_runs" / f"{run}.missing.cache.json"


def command_prepare(args: argparse.Namespace) -> None:
    started = time.time()
    manifest, selected, panels = load_manifest(args.manifest)
    payloads, available = load_sources(selected)
    selected_set = set(selected)

    dataset_report = write_selected_dataset(selected, panels, args.runtime_root / "selected_train.parquet")
    report: dict[str, Any] = {
        "version": 1,
        "created_unix": time.time(),
        "selection_manifest": str(args.manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.manifest),
        "selected_keyset_sha256": keyset_sha256(selected_set),
        "selected_rows": N_SELECTED,
        "dataset": dataset_report,
        "runs": {},
    }
    missing_sets: dict[str, set[str]] = {}
    for run in RUN_SPECS:
        items = available[run]
        missing = selected_set - set(items)
        missing_sets[run] = missing
        partial_payload = {
            "metadata": copy.deepcopy(payloads[run]["metadata"]),
            "items": {key: items[key] for key in sorted(items, key=int)},
            "subset20k": {
                "status": "AVAILABLE_ONLY_NOT_TRAIN_READY_FOR_20480",
                "selected_rows": N_SELECTED,
                "available_rows": len(items),
                "missing_rows": len(missing),
                "selected_keyset_sha256": keyset_sha256(selected_set),
                "available_keyset_sha256": keyset_sha256(set(items)),
                "missing_keyset_sha256": keyset_sha256(missing),
            },
        }
        path = available_cache_path(args.runtime_root, run)
        atomic_json(partial_payload, path)
        missing_path = HERE / "generation_manifests" / run / "missing_indices.json"
        atomic_json(
            {
                "version": 1,
                "run": run,
                "n_missing": len(missing),
                "indices": [int(key) for key in sorted(missing, key=int)],
                "keyset_sha256": keyset_sha256(missing),
            },
            missing_path,
            pretty=True,
        )
        shards = make_shards(run, missing, available, args.runtime_root)
        report["runs"][run] = {
            "source_revision": RUN_SPECS[run]["revision"],
            "source_sha256": RUN_SPECS[run]["sha256"],
            "available_rows": len(items),
            "missing_rows": len(missing),
            "available_cache": str(path.resolve()),
            "available_cache_sha256": sha256_file(path),
            "shards": [
                {
                    "shard": item["shard"],
                    "rows": item["row_count"],
                    "estimated_token_cost": item["estimated_token_cost"],
                    "no_observed_draws": item["no_observed_draw_cost_assigned_full_budget"],
                    "keyset_sha256": item["keyset_sha256"],
                }
                for item in shards
            ],
        }

    report["coverage_intersections"] = {
        "all_three_available": len(selected_set - set().union(*missing_sets.values())),
        "any_available": len(selected_set - set.intersection(*missing_sets.values())),
        "none_available": len(set.intersection(*missing_sets.values())),
        "missing_any_run": len(set().union(*missing_sets.values())),
        "run1_run2_missing": len(missing_sets["run1"] & missing_sets["run2"]),
        "run1_run3_missing": len(missing_sets["run1"] & missing_sets["run3"]),
        "run2_run3_missing": len(missing_sets["run2"] & missing_sets["run3"]),
    }
    report["seconds"] = time.time() - started
    atomic_json(report, HERE / "anchor_coverage.json", pretty=True)
    print(json.dumps(report, indent=2, sort_keys=True))


def load_missing_manifest(run: str) -> tuple[dict[str, Any], set[str]]:
    path = HERE / "generation_manifests" / run / "missing_indices.json"
    payload = load_json(path)
    if payload.get("run") != run:
        raise RuntimeError(f"invalid missing-manifest identity: {path}")
    raw_indices = payload.get("indices", [])
    keys = [str(int(value)) for value in raw_indices]
    unique = set(keys)
    if len(keys) != len(unique) or len(keys) != int(payload.get("n_missing", -1)):
        raise RuntimeError(f"duplicate or mismatched missing rows: {path}")
    if keyset_sha256(unique) != payload.get("keyset_sha256"):
        raise RuntimeError(f"missing-manifest keyset hash mismatch: {path}")
    return payload, unique


def validate_generation_plan(manifest_path: Path, runtime_root: Path) -> dict[str, Any]:
    _, selected, _ = load_manifest(manifest_path)
    _, available = load_sources(selected)
    selected_set = set(selected)
    missing_sets: dict[str, set[str]] = {}
    run_reports: dict[str, Any] = {}

    expected_generation = {
        "model": "Qwen/Qwen3-4B",
        "enable_thinking": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_prompt_tokens": 2048,
        "max_response_tokens": 32768,
        "tensor_model_parallel_size": 1,
        "data_shards": 4,
    }
    seed_bases = {"run1": 11000, "run2": 22000, "run3": 33000}

    for run in sorted(RUN_SPECS):
        actual_missing = selected_set - set(available[run])
        declared, declared_missing = load_missing_manifest(run)
        if declared_missing != actual_missing:
            raise RuntimeError(
                f"{run}: missing manifest differs from pinned-source coverage: "
                f"declared={len(declared_missing)} actual={len(actual_missing)}"
            )
        if len(actual_missing) != EXPECTED_MISSING_ROWS[run]:
            raise RuntimeError(f"{run}: expected {EXPECTED_MISSING_ROWS[run]} missing rows, got {len(actual_missing)}")
        agent_workers = GENERATION_AGENT_WORKERS[run]
        batch_rows = GENERATION_BATCH_ROWS[run]
        if batch_rows % agent_workers:
            raise RuntimeError(f"{run}: batch size {batch_rows} is not divisible by {agent_workers} workers")
        final_rows = len(actual_missing) % batch_rows
        final_padding = 0
        if final_rows:
            final_padding = (agent_workers - final_rows % agent_workers) % agent_workers
        if final_padding > 1:
            raise RuntimeError(f"{run}: final batch would add {final_padding} padding requests")

        shard_union: set[str] = set()
        shard_reports = []
        for shard in range(N_SHARDS):
            shard_manifest = load_shard_manifest(run, shard)
            shard_keys = {str(int(value)) for value in shard_manifest["indices"]}
            overlap = shard_union & shard_keys
            if overlap:
                raise RuntimeError(f"{run}: static shard manifests overlap at {sorted(overlap, key=int)[:5]}")
            shard_union.update(shard_keys)
            generation = shard_manifest.get("generation", {})
            mismatches = {
                key: (generation.get(key), value)
                for key, value in expected_generation.items()
                if generation.get(key) != value
            }
            expected_seed = seed_bases[run] + shard
            if generation.get("sample_seed") != expected_seed:
                mismatches["sample_seed"] = (generation.get("sample_seed"), expected_seed)
            if mismatches:
                raise RuntimeError(f"{run}/shard{shard}: generation manifest mismatch: {mismatches}")
            shard_reports.append(
                {
                    "shard": shard,
                    "rows": len(shard_keys),
                    "estimated_token_cost": int(shard_manifest["estimated_token_cost"]),
                    "keyset_sha256": shard_manifest["keyset_sha256"],
                }
            )
        if shard_union != declared_missing:
            raise RuntimeError(f"{run}: four static shard manifests do not cover the exact missing set")

        missing_sets[run] = actual_missing
        run_reports[run] = {
            "available_rows": len(available[run]),
            "missing_anchor_draws": len(actual_missing),
            "individual_generation_completions": 2 * len(actual_missing),
            "agent_workers": agent_workers,
            "rows_per_agent_worker": math.ceil(len(actual_missing) / agent_workers),
            "precompute_batch_rows": batch_rows,
            "final_batch_rows": final_rows or batch_rows,
            "final_padding_requests": final_padding,
            "submitted_anchor_draws": len(actual_missing) + final_padding,
            "missing_manifest": str((HERE / "generation_manifests" / run / "missing_indices.json").resolve()),
            "missing_keyset_sha256": declared["keyset_sha256"],
            "one_shot_output": str(generated_run_cache_path(runtime_root, run).resolve()),
            "static_balance_audit_shards": shard_reports,
        }

    missing_union = set().union(*missing_sets.values())
    missing_draws = sum(len(keys) for keys in missing_sets.values())
    availability_counts = collections.Counter(sum(key in available[run] for run in RUN_SPECS) for key in selected_set)
    draws_needed_counts = {str(draws_needed): availability_counts[3 - draws_needed] for draws_needed in range(4)}
    if len(missing_union) != EXPECTED_MISSING_UNION:
        raise RuntimeError(
            f"expected {EXPECTED_MISSING_UNION} unique inputs missing at least one draw, got {len(missing_union)}"
        )
    if missing_draws != EXPECTED_MISSING_DRAWS:
        raise RuntimeError(f"expected {EXPECTED_MISSING_DRAWS} missing anchor draws, got {missing_draws}")
    if sum(int(draws) * count for draws, count in draws_needed_counts.items()) != missing_draws:
        raise RuntimeError("availability-pattern accounting does not reproduce missing draw total")

    report = {
        "version": 1,
        "selected_inputs": N_SELECTED,
        "strict_n3_rows_after_success": N_SELECTED,
        "unique_inputs_requiring_generation": len(missing_union),
        "anchor_draws_to_generate": missing_draws,
        "full_generation_completions": 2 * missing_draws,
        "vllm_prompt_logprob_scoring_calls": 2 * missing_draws,
        "rows_by_additional_draws_needed": draws_needed_counts,
        "runs": run_reports,
        "execution": {
            "physical_gpus": [4, 5, 6, 7],
            "vllm_replicas": 4,
            "tensor_model_parallel_size": 1,
            "routing": "global_least_inflight",
            "max_prompt_tokens": 2048,
            "max_completion_tokens": 32768,
            "sampling": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 0.0,
                "thinking": True,
            },
            "model_level_empty_response_retries": 0,
            "launcher_retries": 0,
            "campaign_retries": 0,
        },
    }
    if report["full_generation_completions"] != EXPECTED_GENERATED_COMPLETIONS:
        raise RuntimeError("individual completion accounting changed")
    return report


def command_validate_plan(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            validate_generation_plan(args.manifest, args.runtime_root),
            indent=2,
            sort_keys=True,
        )
    )


def validate_generated_run(run: str, runtime_root: Path) -> dict[str, Any]:
    _, wanted = load_missing_manifest(run)
    output = generated_run_cache_path(runtime_root, run)
    payload = load_json(output)
    validate_metadata(payload, EXPECTED_GENERATION_METADATA, f"{run}/one-shot")
    metadata = payload.get("metadata", {})
    if int(metadata.get("train_sample_count", -1)) != len(wanted):
        raise RuntimeError(
            f"{run}: generated metadata train_sample_count={metadata.get('train_sample_count')} != {len(wanted)}"
        )
    if int(metadata.get("response_length", -1)) != 32768:
        raise RuntimeError(f"{run}: generated response_length is not 32768")
    items = payload.get("items", {})
    if not isinstance(items, dict) or set(items) != wanted:
        item_keys = set(items) if isinstance(items, dict) else set()
        raise RuntimeError(
            f"{run}: one-shot generated keys differ: missing={len(wanted - item_keys)}, extra={len(item_keys - wanted)}"
        )
    not_ready = [key for key, item in items.items() if not generated_item_train_ready(item)]
    if not_ready:
        raise RuntimeError(f"{run}: {len(not_ready)} one-shot rows are not train-ready")
    return {
        "run": run,
        "rows": len(items),
        "full_generation_completions": 2 * len(items),
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "layout": {
            "physical_gpus": [4, 5, 6, 7],
            "vllm_replicas": 4,
            "tensor_model_parallel_size": 1,
            "routing": "global_least_inflight",
        },
    }


def audit_generated_run(run: str, runtime_root: Path) -> dict[str, Any]:
    _, wanted = load_missing_manifest(run)
    output = generated_run_cache_path(runtime_root, run)
    payload = load_json(output)
    validate_metadata(payload, EXPECTED_GENERATION_METADATA, f"{run}/one-shot audit")
    metadata = payload.get("metadata", {})
    if int(metadata.get("train_sample_count", -1)) != len(wanted):
        raise RuntimeError(f"{run}: audited train_sample_count does not match manifest")
    if int(metadata.get("response_length", -1)) != 32768:
        raise RuntimeError(f"{run}: audited response_length is not 32768")
    items = payload.get("items", {})
    if not isinstance(items, dict) or set(items) != wanted:
        item_keys = set(items) if isinstance(items, dict) else set()
        raise RuntimeError(
            f"{run}: audited keys differ: missing={len(wanted - item_keys)}, extra={len(item_keys - wanted)}"
        )

    issue_counts: collections.Counter[str] = collections.Counter()
    train_ready_rows = 0
    for item in items.values():
        if generated_item_train_ready(item):
            train_ready_rows += 1
            continue
        if not isinstance(item, dict):
            issue_counts["non_mapping_item"] += 1
            continue
        if not item.get("y0"):
            issue_counts["missing_y0"] += 1
        if not item.get("y1"):
            issue_counts["missing_y1"] += 1
        if not complete_item(item):
            issue_counts["incomplete_score"] += 1
        else:
            issue_counts["malformed_count_ppl_or_length"] += 1

    return {
        "run": run,
        "exact_manifest_keys_recorded": True,
        "anchor_draws_attempted": len(items),
        "full_generation_completions_attempted": 2 * len(items),
        "train_ready_rows": train_ready_rows,
        "not_train_ready_rows": len(items) - train_ready_rows,
        "strictly_train_ready": train_ready_rows == len(items),
        "issue_counts": dict(sorted(issue_counts.items())),
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
    }


def command_audit_generated_run(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            audit_generated_run(args.run, args.runtime_root),
            indent=2,
            sort_keys=True,
        )
    )


def command_validate_generated_run(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            validate_generated_run(args.run, args.runtime_root),
            indent=2,
            sort_keys=True,
        )
    )


def load_shard_manifest(run: str, shard: int) -> dict[str, Any]:
    path = HERE / "generation_manifests" / run / f"shard{shard}.indices.json"
    payload = load_json(path)
    if payload.get("run") != run or int(payload.get("shard", -1)) != shard:
        raise RuntimeError(f"invalid shard identity: {path}")
    keys = {str(int(value)) for value in payload.get("indices", [])}
    if len(keys) != int(payload.get("row_count", -1)):
        raise RuntimeError(f"duplicate or mismatched shard rows: {path}")
    if keyset_sha256(keys) != payload.get("keyset_sha256"):
        raise RuntimeError(f"shard keyset hash mismatch: {path}")
    return payload


def validate_generated_shard(run: str, shard: int) -> dict[str, Any]:
    manifest = load_shard_manifest(run, shard)
    output = Path(manifest["output_cache"])
    payload = load_json(output)
    validate_metadata(payload, EXPECTED_GENERATION_METADATA, f"{run}/shard{shard}")
    items = payload.get("items", {})
    wanted = {str(int(value)) for value in manifest["indices"]}
    if set(items) != wanted:
        raise RuntimeError(
            f"{run}/shard{shard}: generated keys differ: "
            f"missing={len(wanted - set(items))}, extra={len(set(items) - wanted)}"
        )
    incomplete = [key for key, item in items.items() if not complete_item(item)]
    if incomplete:
        raise RuntimeError(f"{run}/shard{shard}: {len(incomplete)} incomplete generated rows")
    return {"rows": len(items), "path": str(output.resolve()), "sha256": sha256_file(output)}


def command_validate_shard(args: argparse.Namespace) -> None:
    print(json.dumps(validate_generated_shard(args.run, args.shard), indent=2, sort_keys=True))


def command_finalize_run(args: argparse.Namespace) -> None:
    _, selected, _ = load_manifest(args.manifest)
    payloads, available = load_sources(selected)
    selected_set = set(selected)
    missing = selected_set - set(available[args.run])
    generated: dict[str, Any] = {}
    generation_records = []
    if args.input_mode == "run":
        record = validate_generated_run(args.run, args.runtime_root)
        generation_records.append(record)
        generated.update(load_json(Path(record["path"]))["items"])
    else:
        for shard in range(N_SHARDS):
            record = validate_generated_shard(args.run, shard)
            generation_records.append(record)
            shard_payload = load_json(Path(record["path"]))
            overlap = set(generated) & set(shard_payload["items"])
            if overlap:
                raise RuntimeError(f"generated shards overlap: {sorted(overlap, key=int)[:5]}")
            generated.update(shard_payload["items"])
    if set(generated) != missing:
        raise RuntimeError(
            f"{args.run}: generated union != missing set: missing={len(missing)}, generated={len(generated)}"
        )
    merged = {**available[args.run], **generated}
    if set(merged) != selected_set or any(not complete_item(item) for item in merged.values()):
        raise RuntimeError(f"{args.run}: final merged cache is not exactly complete")
    output_payload = {
        "metadata": copy.deepcopy(payloads["run1"]["metadata"]),
        "items": {key: merged[key] for key in sorted(merged, key=int)},
        "subset20k_completion": {
            "version": 1,
            "run": args.run,
            "selected_rows": N_SELECTED,
            "selected_keyset_sha256": keyset_sha256(selected_set),
            "source_revision": RUN_SPECS[args.run]["revision"],
            "source_sha256": RUN_SPECS[args.run]["sha256"],
            "source_complete_selected_rows": len(available[args.run]),
            "generated_missing_rows": len(generated),
            "generated_keyset_sha256": keyset_sha256(set(generated)),
            "fallback_max_response_tokens": 32768,
            "data_shards": N_SHARDS,
            "generation_layout": args.input_mode,
            "vllm_replicas": N_SHARDS,
            "tensor_model_parallel_size": 1,
            "generation_artifacts": generation_records,
        },
    }
    output = complete_cache_path(args.runtime_root, args.run)
    atomic_json(output_payload, output)
    reloaded = load_json(output)
    validate_metadata(reloaded, EXPECTED_SOURCE_METADATA, f"final {args.run}")
    if set(reloaded.get("items", {})) != selected_set:
        raise RuntimeError("serialized final key set changed")
    print(f"{args.run}: wrote {N_SELECTED:,} complete rows; sha256={sha256_file(output)} path={output}")


def validate_complete_run(run: str, runtime_root: Path, manifest_path: Path) -> dict[str, Any]:
    _, selected, _ = load_manifest(manifest_path)
    selected_set = set(selected)
    output = complete_cache_path(runtime_root, run)
    payload = load_json(output)
    validate_metadata(payload, EXPECTED_SOURCE_METADATA, f"final {run}")
    items = payload.get("items", {})
    if not isinstance(items, dict) or set(items) != selected_set:
        item_keys = set(items) if isinstance(items, dict) else set()
        raise RuntimeError(
            f"{run}: final keys differ from fixed 20K selection: "
            f"missing={len(selected_set - item_keys)}, extra={len(item_keys - selected_set)}"
        )
    incomplete = [key for key, item in items.items() if not complete_item(item)]
    if incomplete:
        raise RuntimeError(f"{run}: final cache has {len(incomplete)} incomplete rows")
    completion = payload.get("subset20k_completion", {})
    if completion.get("run") != run:
        raise RuntimeError(f"{run}: final cache has invalid completion provenance")
    if int(completion.get("selected_rows", -1)) != N_SELECTED:
        raise RuntimeError(f"{run}: final cache selected-row count changed")
    if int(completion.get("generated_missing_rows", -1)) != EXPECTED_MISSING_ROWS[run]:
        raise RuntimeError(f"{run}: final cache generated-row count changed")
    return {
        "run": run,
        "rows": len(items),
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "generated_missing_rows": int(completion["generated_missing_rows"]),
    }


def command_validate_complete(args: argparse.Namespace) -> None:
    runs = {run: validate_complete_run(run, args.runtime_root, args.manifest) for run in sorted(RUN_SPECS)}
    print(
        json.dumps(
            {
                "strict_n3_rows": N_SELECTED,
                "completed_anchor_draws": N_SELECTED * len(RUN_SPECS),
                "generated_anchor_draws": EXPECTED_MISSING_DRAWS,
                "generated_full_completions": EXPECTED_GENERATED_COMPLETIONS,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    result.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT_DEFAULT)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("validate-plan")
    audited = commands.add_parser("audit-generated-run")
    audited.add_argument("--run", choices=sorted(RUN_SPECS), required=True)
    generated = commands.add_parser("validate-generated-run")
    generated.add_argument("--run", choices=sorted(RUN_SPECS), required=True)
    validate = commands.add_parser("validate-shard")
    validate.add_argument("--run", choices=sorted(RUN_SPECS), required=True)
    validate.add_argument("--shard", type=int, choices=range(N_SHARDS), required=True)
    finalize = commands.add_parser("finalize-run")
    finalize.add_argument("--run", choices=sorted(RUN_SPECS), required=True)
    finalize.add_argument("--input-mode", choices=("run", "shards"), default="run")
    commands.add_parser("validate-complete")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "validate-plan":
        command_validate_plan(args)
    elif args.command == "audit-generated-run":
        command_audit_generated_run(args)
    elif args.command == "validate-generated-run":
        command_validate_generated_run(args)
    elif args.command == "validate-shard":
        command_validate_shard(args)
    elif args.command == "finalize-run":
        command_finalize_run(args)
    elif args.command == "validate-complete":
        command_validate_complete(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
