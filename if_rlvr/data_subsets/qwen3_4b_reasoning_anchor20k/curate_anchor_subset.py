#!/usr/bin/env python3
"""Select a nested, representative 20,480-row Qwen3-4B anchor subset.

The first 4,096 rows are the already validated subset used by the A/B/N=3
experiments.  The remaining 16,384 rows are selected from the same 93,882-row
effective training population to minimize multivariate distance from the full
set while correcting the fixed panel's small residual differences.

Hard allocation uses hierarchical strata over anchor availability, anchor
hygiene, cross-run flips, constraint count, multi-run policy difficulty, and
multi-run full-compliance behavior.  Rerandomization and within-stratum swap
polishing then match continuous input/anchor geometry plus constraint-ID,
family, co-occurrence, source-domain, and constraint-signature marginals.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


ROOT = Path(__file__).resolve().parents[3]
N_SUBSET = 20_480
N_FIXED = 4_096
SEED = 20_260_819
RERANDOMIZED_DRAWS = 500
SWAP_PROPOSALS = 30_000
GRID = 512
MIN_EXPECTED_CELL = 5.0

# Pre-registered whole-subset acceptance thresholds.  The two-sample 5% KS
# critical value at n=20,480 vs N=93,882 is about 0.0107, so 0.006 is strict.
ACC_MAX_KS = 0.006
ACC_TV_ID = 0.0015
ACC_TV_FAMILY = 0.0010
ACC_TV_PAIR = 0.0050
ACC_TV_SOURCE = 0.0050
ACC_TV_SIGNATURE = 0.0100
ACC_TV_AVAILABILITY_PATTERN = 0.0050

W_ID = 8.0
W_FAMILY = 3.0
W_PAIR = 4.0
W_SOURCE = 5.0
W_SIGNATURE = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bucket(key: str) -> str:
    """Collapse per-example IDs into 55 stable source/domain buckets."""
    if key.startswith("ai2-adapt-dev/"):
        value = key.split("/", 1)[1]
        value = re.sub(r"_[0-9]+$", "", value)
        value = re.sub(r"_[a-z0-9]{16,}$", "", value)
        return f"ai2/{value}"
    for prefix in ("personas_math_easy_", "personahub_", "oasst1_", "hard_coded_"):
        if key.startswith(prefix):
            return prefix.rstrip("_")
    if key.startswith("science."):
        return key.rsplit(".", 1)[0]
    if re.fullmatch(r"[a-z0-9]{5,12}", key):
        return "opaque_short_id"
    return re.split(r"[_./]", key, maxsplit=1)[0] or "unknown"


def one_hot(labels: list[str], levels: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if levels is None:
        levels = sorted(set(labels))
    position = {label: index for index, label in enumerate(levels)}
    matrix = np.zeros((len(labels), len(levels)), dtype=np.int8)
    for row, label in enumerate(labels):
        matrix[row, position[label]] = 1
    return matrix, levels


def tv_from_counts(sample: np.ndarray, full: np.ndarray) -> float:
    if sample.sum() <= 0 or full.sum() <= 0:
        return 0.0
    return float(0.5 * np.abs(sample / sample.sum() - full / full.sum()).sum())


def constrained_allocation(
    strata_members: dict[str, np.ndarray], fixed_mask: np.ndarray, target_size: int
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Nearest-L2 integer allocation subject to inclusion of all fixed rows."""
    population = sum(len(values) for values in strata_members.values())
    names = list(strata_members)
    fixed_counts = {
        name: int(fixed_mask[values].sum()) for name, values in strata_members.items()
    }
    final = dict(fixed_counts)
    ideal = {name: len(strata_members[name]) * target_size / population for name in names}
    remaining = target_size - sum(final.values())
    if remaining < 0:
        raise RuntimeError("fixed panel exceeds target subset size")

    # Adding one row changes squared allocation error by 2*(count-ideal)+1.
    import heapq

    heap: list[tuple[float, str]] = []
    for name in names:
        if final[name] < len(strata_members[name]):
            heapq.heappush(heap, (2.0 * (final[name] - ideal[name]) + 1.0, name))
    for _ in range(remaining):
        if not heap:
            raise RuntimeError("stratified allocation exhausted population capacity")
        _, name = heapq.heappop(heap)
        final[name] += 1
        if final[name] < len(strata_members[name]):
            heapq.heappush(heap, (2.0 * (final[name] - ideal[name]) + 1.0, name))
    added = {name: final[name] - fixed_counts[name] for name in names}
    return final, added, ideal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-audit", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=RERANDOMIZED_DRAWS)
    parser.add_argument("--swaps", type=int, default=SWAP_PROPOSALS)
    args = parser.parse_args()

    started = time.time()
    frame = pd.read_parquet(args.audit)
    if len(frame) != 93_882 or not frame["index"].is_unique:
        raise RuntimeError(f"unexpected audit frame: rows={len(frame)}, unique={frame['index'].is_unique}")
    frame = frame.sort_values("index").reset_index(drop=True)
    population = len(frame)
    position_of_index = {int(value): row for row, value in enumerate(frame["index"])}

    with args.fixed_manifest.open("r", encoding="utf-8") as stream:
        fixed_payload = json.load(stream)
    fixed_indices = {int(row["index"]) for row in fixed_payload["rows"]}
    if len(fixed_indices) != N_FIXED or not fixed_indices <= set(position_of_index):
        raise RuntimeError("fixed 4,096-row manifest is incomplete or outside the audit frame")
    fixed_mask = frame["index"].isin(fixed_indices).to_numpy()

    id_lists = [json.loads(value) for value in frame["constraint_ids_json"]]
    family_lists = [json.loads(value) for value in frame["constraint_families_json"]]
    all_ids = sorted({value for values in id_lists for value in values})
    all_families = sorted({value for values in family_lists for value in values})
    id_position = {value: index for index, value in enumerate(all_ids)}
    family_position = {value: index for index, value in enumerate(all_families)}
    ID = np.zeros((population, len(all_ids)), dtype=np.int8)
    FAMILY = np.zeros((population, len(all_families)), dtype=np.int8)
    for row, (ids, families) in enumerate(zip(id_lists, family_lists, strict=True)):
        for value in ids:
            ID[row, id_position[value]] += 1
        for value in families:
            FAMILY[row, family_position[value]] += 1

    pair_counter: Counter[tuple[str, str]] = Counter()
    for values in id_lists:
        pair_counter.update(itertools.combinations(sorted(set(values)), 2))
    top_pairs = [pair for pair, _ in pair_counter.most_common(100)]
    pair_position = {value: index for index, value in enumerate(top_pairs)}
    PAIR = np.zeros((population, len(top_pairs)), dtype=np.int8)
    for row, values in enumerate(id_lists):
        for pair in itertools.combinations(sorted(set(values)), 2):
            column = pair_position.get(pair)
            if column is not None:
                PAIR[row, column] += 1

    source_labels = [source_bucket(value) for value in frame["dataset_key"]]
    SOURCE, source_levels = one_hot(source_labels)
    signature_counts = Counter(frame["constraint_signature"])
    top_signatures = {value for value, _ in signature_counts.most_common(200)}
    signature_labels = [value if value in top_signatures else "__OTHER__" for value in frame["constraint_signature"]]
    SIGNATURE, signature_levels = one_hot(signature_labels)

    available = np.column_stack([frame[f"run{run}_available"].to_numpy(bool) for run in (1, 2, 3)])
    m0 = np.column_stack([frame[f"run{run}_m0"].to_numpy(float) for run in (1, 2, 3)])
    m1 = np.column_stack([frame[f"run{run}_m1"].to_numpy(float) for run in (1, 2, 3)])
    y0_length = np.column_stack([frame[f"run{run}_y0_tokens"].to_numpy(float) for run in (1, 2, 3)])
    y1_length = np.column_stack([frame[f"run{run}_y1_tokens"].to_numpy(float) for run in (1, 2, 3)])
    ifscore = np.column_stack([frame[f"run{run}_ifscore"].to_numpy(float) for run in (1, 2, 3)])
    allsat = np.column_stack([frame[f"run{run}_allsat"].to_numpy(bool) for run in (1, 2, 3)]) & available
    flipped = np.column_stack([frame[f"run{run}_flipped"].to_numpy(bool) for run in (1, 2, 3)]) & available
    availability_count = available.sum(axis=1)
    availability_pattern = sum(available[:, column].astype(np.int8) << column for column in range(3))
    flip_count = flipped.sum(axis=1)
    allsat_count = allsat.sum(axis=1)

    score_sum = np.nansum(ifscore, axis=1)
    row_ifscore = np.divide(
        score_sum,
        availability_count,
        out=np.full(population, np.nan),
        where=availability_count > 0,
    )
    observed = np.isfinite(row_ifscore)
    id_difficulty = (ID.T @ np.where(observed, row_ifscore, 0.0)) / np.maximum(ID.T @ observed.astype(float), 1.0)
    difficulty = (ID @ id_difficulty) / np.maximum(ID.sum(axis=1), 1)
    difficulty_q = np.asarray(
        pd.qcut(difficulty, 5, labels=False, duplicates="drop"), dtype=np.int8
    )

    anchor_m0 = np.divide(
        np.nansum(m0, axis=1), availability_count, out=np.full(population, np.nan), where=availability_count > 0
    )
    anchor_m1 = np.divide(
        np.nansum(m1, axis=1), availability_count, out=np.full(population, np.nan), where=availability_count > 0
    )
    mean_y0_length = np.divide(
        np.where(available, y0_length, 0).sum(axis=1),
        availability_count,
        out=np.full(population, np.nan),
        where=availability_count > 0,
    )
    mean_y1_length = np.divide(
        np.where(available, y1_length, 0).sum(axis=1),
        availability_count,
        out=np.full(population, np.nan),
        where=availability_count > 0,
    )

    def row_std(values: np.ndarray) -> np.ndarray:
        result = np.zeros(population)
        for row in np.where(availability_count >= 2)[0]:
            result[row] = np.std(values[row, available[row]], ddof=1)
        return result

    m0_std = row_std(m0)
    m1_std = row_std(m1)
    all3 = availability_count == 3
    ci95_lower = np.full(population, np.nan)
    ci95_upper = np.full(population, np.nan)
    ci95_lower[all3] = anchor_m0[all3] - 2.484137711719546 * m0_std[all3]
    ci95_upper[all3] = anchor_m1[all3] + 2.484137711719546 * m1_std[all3]

    replicate_family = np.array(
        [bool(set(values) & {"copy", "combination", "new"}) for values in family_lists]
    )
    # Canonical defects use run1 when present, otherwise run2 then run3.
    canonical_run = np.argmax(available, axis=1)
    canonical_run[availability_count == 0] = -1
    canonical_flags: dict[str, np.ndarray] = {}
    for flag in ("short_y0", "short_y1", "loop_y0", "loop_y1", "duplicate", "cot_marker", "allsat"):
        values = np.zeros(population, dtype=bool)
        for run in range(3):
            mask = canonical_run == run
            values[mask] = frame.loc[mask, f"run{run+1}_{flag}"].to_numpy(bool)
        canonical_flags[flag] = values
    suspect_loop_y1 = canonical_flags["loop_y1"] & ~canonical_flags["allsat"] & ~replicate_family
    defect = np.select(
        [
            availability_count == 0,
            canonical_flags["short_y0"],
            canonical_flags["short_y1"],
            suspect_loop_y1,
            canonical_flags["loop_y0"],
            canonical_flags["duplicate"],
            canonical_flags["cot_marker"],
        ],
        ["no_anchor", "short_y0", "short_y1", "suspect_loop_y1", "loop_y0", "duplicate", "cot_marker"],
        default="clean",
    )
    allsat_bucket = np.select(
        [availability_count == 0, allsat_count == 0, allsat_count == availability_count],
        ["no_anchor", "none", "all"],
        default="some",
    )

    # Hierarchical exact strata. Availability pattern is never merged away.
    detailed = list(
        zip(
            availability_pattern,
            defect,
            flip_count,
            frame["n_constraints"].to_numpy(int),
            difficulty_q,
            allsat_bucket,
            strict=True,
        )
    )
    candidates = [
        detailed,
        [value[:5] for value in detailed],
        [value[:4] for value in detailed],
        [value[:3] for value in detailed],
        [(value[0], value[1]) for value in detailed],
    ]
    counts = [Counter(values) for values in candidates]
    stratum_key: list[tuple[Any, ...]] = []
    for row in range(population):
        chosen: tuple[Any, ...] | None = None
        for level, values in enumerate(candidates):
            key = values[row]
            if counts[level][key] * N_SUBSET / population >= MIN_EXPECTED_CELL:
                chosen = (f"L{6-level}",) + tuple(key)
                break
        if chosen is None:
            chosen = ("L2",) + tuple(candidates[-1][row])
        stratum_key.append(chosen)
    stratum_labels = [repr(value) for value in stratum_key]
    strata_members = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in pd.Series(np.arange(population)).groupby(stratum_labels, sort=False).groups.items()
    }
    final_allocation, add_allocation, ideal_allocation = constrained_allocation(
        strata_members, fixed_mask, N_SUBSET
    )
    print(
        f"population={population:,} fixed={fixed_mask.sum():,} strata={len(strata_members):,} "
        f"add={sum(add_allocation.values()):,}",
        flush=True,
    )

    # Continuous balancing variables. NaNs receive a dedicated low sentinel;
    # exact availability strata ensure this never disguises missingness.
    V = {
        "prompt_tokens": frame["prompt_tokens"].to_numpy(float),
        "x_prompt_tokens": frame["x_prompt_tokens"].to_numpy(float),
        "constraint_tokens": frame["constraint_tokens"].to_numpy(float),
        "log_x_chars": np.log1p(frame["x_chars"].to_numpy(float)),
        "log_constraint_chars": np.log1p(frame["constraint_chars"].to_numpy(float)),
        "log_x_words": np.log1p(frame["x_words"].to_numpy(float)),
        "x_ascii_ratio": frame["x_ascii_ratio"].to_numpy(float),
        "x_cjk_ratio": frame["x_cjk_ratio"].to_numpy(float),
        "x_cyrillic_ratio": frame["x_cyrillic_ratio"].to_numpy(float),
        "x_arabic_ratio": frame["x_arabic_ratio"].to_numpy(float),
        "x_devanagari_ratio": frame["x_devanagari_ratio"].to_numpy(float),
        "x_hangul_ratio": frame["x_hangul_ratio"].to_numpy(float),
        "difficulty": difficulty,
        "observed_ifscore": row_ifscore,
        "anchor_m0_mean": anchor_m0,
        "anchor_m1_mean": anchor_m1,
        "anchor_center_width": anchor_m1 - anchor_m0,
        "log_anchor_y0_tokens": np.log1p(mean_y0_length),
        "log_anchor_y1_tokens": np.log1p(mean_y1_length),
        "log_anchor_length_ratio": np.log((mean_y1_length + 1) / (mean_y0_length + 1)),
        "anchor_m0_std": m0_std,
        "anchor_m1_std": m1_std,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
    }
    clean_V: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}
    full_cdf: dict[str, np.ndarray] = {}
    bins: dict[str, np.ndarray] = {}
    for name, raw in V.items():
        finite = raw[np.isfinite(raw)]
        sentinel = (finite.min() - max(np.ptp(finite), 1.0)) if len(finite) else -1.0
        value = np.where(np.isfinite(raw), raw, sentinel)
        clean_V[name] = value
        unique = np.unique(value)
        grid = unique if len(unique) <= GRID else np.unique(np.quantile(value, np.linspace(0, 1, GRID)))
        grids[name] = grid
        full_cdf[name] = np.searchsorted(np.sort(value), grid, side="right") / population
        bins[name] = np.searchsorted(grid, value, side="left").astype(np.int32)

    full_counts = {
        "id": ID.sum(axis=0).astype(float),
        "family": FAMILY.sum(axis=0).astype(float),
        "pair": PAIR.sum(axis=0).astype(float),
        "source": SOURCE.sum(axis=0).astype(float),
        "signature": SIGNATURE.sum(axis=0).astype(float),
    }

    def score_parts(pick: np.ndarray) -> tuple[float, dict[str, float], dict[str, float], float]:
        ks: dict[str, float] = {}
        for name in V:
            histogram = np.bincount(bins[name][pick], minlength=len(grids[name]))
            ks[name] = float(np.max(np.abs(np.cumsum(histogram) / len(pick) - full_cdf[name])))
        tv = {
            "id": tv_from_counts(ID[pick].sum(axis=0), full_counts["id"]),
            "family": tv_from_counts(FAMILY[pick].sum(axis=0), full_counts["family"]),
            "pair": tv_from_counts(PAIR[pick].sum(axis=0), full_counts["pair"]),
            "source": tv_from_counts(SOURCE[pick].sum(axis=0), full_counts["source"]),
            "signature": tv_from_counts(SIGNATURE[pick].sum(axis=0), full_counts["signature"]),
        }
        sample_id = ID[pick].sum(axis=0).astype(float)
        expected_id = full_counts["id"] / full_counts["id"].sum() * sample_id.sum()
        coverage = float(np.min(sample_id / np.maximum(expected_id, 1e-12)))
        composite = (
            sum(ks.values())
            + W_ID * tv["id"]
            + W_FAMILY * tv["family"]
            + W_PAIR * tv["pair"]
            + W_SOURCE * tv["source"]
            + W_SIGNATURE * tv["signature"]
            + (100.0 if coverage < 0.8 else 0.0)
        )
        return composite, ks, tv, coverage

    fixed_positions = np.flatnonzero(fixed_mask)
    available_by_stratum = {
        name: values[~fixed_mask[values]] for name, values in strata_members.items()
    }
    rng_master = np.random.default_rng(SEED)
    best_pick: np.ndarray | None = None
    best_score = float("inf")
    draw_scores: list[float] = []
    for draw in range(args.draws):
        rng = np.random.default_rng(rng_master.integers(0, 2**63))
        added = [
            rng.choice(available_by_stratum[name], size=count, replace=False)
            for name, count in add_allocation.items()
            if count > 0
        ]
        pick = np.concatenate([fixed_positions, *added])
        if len(pick) != N_SUBSET or len(np.unique(pick)) != N_SUBSET:
            raise RuntimeError("candidate draw is not a unique 20,480-row set")
        composite = score_parts(pick)[0]
        draw_scores.append(composite)
        if composite < best_score:
            best_score, best_pick = composite, pick.copy()
        if (draw + 1) % 100 == 0:
            print(f"rerandomization {draw+1}/{args.draws}: best={best_score:.6f}", flush=True)
    assert best_pick is not None

    # Incremental within-stratum swap polish; fixed panel rows are immutable.
    in_subset = np.zeros(population, dtype=bool)
    in_subset[best_pick] = True
    histograms = {
        name: np.bincount(bins[name][best_pick], minlength=len(grids[name])).astype(np.int32)
        for name in V
    }
    sums = {
        "id": ID[best_pick].sum(axis=0).astype(np.int64),
        "family": FAMILY[best_pick].sum(axis=0).astype(np.int64),
        "pair": PAIR[best_pick].sum(axis=0).astype(np.int64),
        "source": SOURCE[best_pick].sum(axis=0).astype(np.int64),
        "signature": SIGNATURE[best_pick].sum(axis=0).astype(np.int64),
    }

    def state_score() -> float:
        ks_sum = sum(
            float(np.max(np.abs(np.cumsum(histograms[name]) / N_SUBSET - full_cdf[name])))
            for name in V
        )
        tv_id = tv_from_counts(sums["id"], full_counts["id"])
        tv_family = tv_from_counts(sums["family"], full_counts["family"])
        tv_pair = tv_from_counts(sums["pair"], full_counts["pair"])
        tv_source = tv_from_counts(sums["source"], full_counts["source"])
        tv_signature = tv_from_counts(sums["signature"], full_counts["signature"])
        expected_id = full_counts["id"] / full_counts["id"].sum() * sums["id"].sum()
        coverage = float(np.min(sums["id"] / np.maximum(expected_id, 1e-12)))
        return (
            ks_sum
            + W_ID * tv_id
            + W_FAMILY * tv_family
            + W_PAIR * tv_pair
            + W_SOURCE * tv_source
            + W_SIGNATURE * tv_signature
            + (100.0 if coverage < 0.8 else 0.0)
        )

    current = state_score()
    swappable = {
        name: values
        for name, values in available_by_stratum.items()
        if 0 < add_allocation[name] < len(values)
    }
    swap_names = list(swappable)
    swap_weights = np.array([len(swappable[name]) for name in swap_names], dtype=float)
    swap_weights /= swap_weights.sum()
    rng = np.random.default_rng(SEED + 7)
    accepted = 0
    matrices = {"id": ID, "family": FAMILY, "pair": PAIR, "source": SOURCE, "signature": SIGNATURE}
    for proposal in range(args.swaps):
        name = swap_names[rng.choice(len(swap_names), p=swap_weights)]
        members = swappable[name]
        selected = members[in_subset[members]]
        unselected = members[~in_subset[members]]
        if not len(selected) or not len(unselected):
            continue
        old = int(selected[rng.integers(len(selected))])
        new = int(unselected[rng.integers(len(unselected))])
        for variable in V:
            histograms[variable][bins[variable][old]] -= 1
            histograms[variable][bins[variable][new]] += 1
        for key, matrix in matrices.items():
            sums[key] += matrix[new].astype(np.int64) - matrix[old].astype(np.int64)
        candidate = state_score()
        if candidate < current - 1e-12:
            current = candidate
            in_subset[old] = False
            in_subset[new] = True
            accepted += 1
        else:
            for variable in V:
                histograms[variable][bins[variable][old]] += 1
                histograms[variable][bins[variable][new]] -= 1
            for key, matrix in matrices.items():
                sums[key] -= matrix[new].astype(np.int64) - matrix[old].astype(np.int64)
        if (proposal + 1) % 5_000 == 0:
            print(
                f"swap polish {proposal+1}/{args.swaps}: accepted={accepted} score={current:.6f}",
                flush=True,
            )

    pick = np.flatnonzero(in_subset)
    if len(pick) != N_SUBSET or not fixed_mask[pick].sum() == N_FIXED:
        raise RuntimeError("final subset lost size or fixed-panel invariants")
    composite, ks, tv, coverage = score_parts(pick)

    # Assign new rows to four 4,096-row panels. Each stratum is spread as evenly
    # as possible, with extremes alternated using a multivariate rank proxy.
    panel = np.full(population, -1, dtype=np.int8)
    panel[fixed_mask] = 0
    panel_counts = np.zeros(5, dtype=int)
    panel_counts[0] = N_FIXED
    stratum_panel_counts: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(SEED + 19)
    rank_proxy = (
        pd.Series(clean_V["prompt_tokens"]).rank(pct=True).to_numpy()
        + pd.Series(clean_V["anchor_m1_mean"]).rank(pct=True).to_numpy()
        + pd.Series(clean_V["observed_ifscore"]).rank(pct=True).to_numpy()
    )
    for name, members in strata_members.items():
        rows = members[in_subset[members] & ~fixed_mask[members]]
        if not len(rows):
            continue
        jitter = rng.random(len(rows)) * 1e-9
        rows = rows[np.argsort(rank_proxy[rows] + jitter)]
        local = np.zeros(5, dtype=int)
        direction = 1
        for row in rows:
            choices = list(range(1, 5))
            if direction < 0:
                choices.reverse()
            chosen = min(choices, key=lambda value: (local[value], panel_counts[value], value if direction > 0 else -value))
            panel[row] = chosen
            local[chosen] += 1
            panel_counts[chosen] += 1
            direction *= -1
        stratum_panel_counts[name] = local
    if panel_counts.tolist() != [N_FIXED] * 5 or np.any(panel[pick] < 0):
        raise RuntimeError(f"panel assignment is not 5 x 4,096: {panel_counts.tolist()}")

    selected = frame.iloc[pick].copy()
    selected["panel"] = ["ABCDE"[value] for value in panel[pick]]
    selected["difficulty_score"] = difficulty[pick]
    selected["difficulty_quintile"] = difficulty_q[pick]
    selected["availability_pattern"] = availability_pattern[pick]
    selected["availability_count"] = availability_count[pick]
    selected["flip_count"] = flip_count[pick]
    selected["allsat_bucket"] = allsat_bucket[pick]
    selected["canonical_defect"] = defect[pick]
    selected["stratum"] = [stratum_labels[value] for value in pick]
    selected["source_bucket"] = [source_labels[value] for value in pick]
    args.selected_audit.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.selected_audit, index=False, compression="zstd")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position in pick[np.argsort(frame.iloc[pick]["index"].to_numpy())]:
        rows.append(
            {
                "index": int(frame.at[position, "index"]),
                "panel": "ABCDE"[panel[position]],
                "stratum": stratum_labels[position],
                "n_constraints": int(frame.at[position, "n_constraints"]),
                "difficulty_quintile": int(difficulty_q[position]),
                "availability_pattern": int(availability_pattern[position]),
                "flip_count": int(flip_count[position]),
                "allsat_bucket": str(allsat_bucket[position]),
                "canonical_defect": str(defect[position]),
                "run1_missing": not bool(available[position, 0]),
                "run2_missing": not bool(available[position, 1]),
                "run3_missing": not bool(available[position, 2]),
            }
        )
    manifest = {
        "version": 1,
        "n": N_SUBSET,
        "seed": SEED,
        "frame_rows": population,
        "fixed_nested_rows": N_FIXED,
        "audit_path": str(args.audit.resolve()),
        "audit_sha256": sha256_file(args.audit),
        "fixed_manifest": str(args.fixed_manifest.resolve()),
        "method": "fixed-4k + hierarchical stratification + rerandomization + within-stratum swaps",
        "panels": {name: sum(row["panel"] == name for row in rows) for name in "ABCDE"},
        "rows": rows,
    }
    manifest_path = args.output_dir / "subset_indices.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, separators=(",", ":"), ensure_ascii=False)
        stream.write("\n")
    indices = sorted(int(frame.at[value, "index"]) for value in pick)
    with (args.output_dir / "train_indices.json").open("w", encoding="utf-8") as stream:
        json.dump({"indices": indices}, stream, separators=(",", ":"))
        stream.write("\n")
    for panel_name in "ABCDE":
        panel_indices = sorted(
            int(frame.at[value, "index"]) for value in pick if "ABCDE"[panel[value]] == panel_name
        )
        with (args.output_dir / f"panel_{panel_name}_indices.json").open("w", encoding="utf-8") as stream:
            json.dump({"indices": panel_indices}, stream, separators=(",", ":"))
            stream.write("\n")

    # ---------------- verification report ----------------
    subset_pattern = np.bincount(availability_pattern[pick], minlength=8)
    full_pattern = np.bincount(availability_pattern, minlength=8)
    pattern_tv = tv_from_counts(subset_pattern, full_pattern)
    checks = {
        "max_continuous_KS": (max(ks.values()), ACC_MAX_KS),
        "TV_constraint_ID": (tv["id"], ACC_TV_ID),
        "TV_constraint_family": (tv["family"], ACC_TV_FAMILY),
        "TV_top100_ID_pairs": (tv["pair"], ACC_TV_PAIR),
        "TV_source_domain": (tv["source"], ACC_TV_SOURCE),
        "TV_top200_signature": (tv["signature"], ACC_TV_SIGNATURE),
        "TV_availability_pattern": (pattern_tv, ACC_TV_AVAILABILITY_PATTERN),
    }
    all_pass = all(value < threshold for value, threshold in checks.values())
    report: list[str] = [
        "# Qwen3-4B representative 20,480-row subset — verification",
        "",
        f"Population: {population:,} trainable rows. Subset: {N_SUBSET:,} rows "
        f"({100*N_SUBSET/population:.2f}%). Nested fixed panel A: {N_FIXED:,} rows.",
        f"Seed: {SEED}. Composite: random median {np.median(draw_scores):.6f} → "
        f"best draw {best_score:.6f} → polished {composite:.6f} ({accepted} swaps).",
        "",
        "## Pre-registered acceptance checks",
        "",
    ]
    for name, (value, threshold) in checks.items():
        report.append(f"- {'PASS' if value < threshold else '**FAIL**'} — {name}: {value:.6f} < {threshold:.6f}")
    report.extend(["", "## Continuous distribution checks", "", "| variable | KS D |", "|---|---:|"])
    for name, value in sorted(ks.items(), key=lambda item: -item[1]):
        report.append(f"| {name} | {value:.6f} |")

    report.extend(
        [
            "",
            "## Anchor availability and geometry",
            "",
            "| statistic | full | subset |",
            "|---|---:|---:|",
        ]
    )
    for run in range(3):
        full_avail = available[:, run]
        sub_avail = available[pick, run]
        pairs = [
            (f"run{run+1} available", full_avail.mean(), sub_avail.mean()),
            (
                f"run{run+1} flip | available",
                flipped[full_avail, run].mean(),
                flipped[pick, run][sub_avail].mean(),
            ),
            (
                f"run{run+1} IFEval score | available",
                np.nanmean(ifscore[:, run]),
                np.nanmean(ifscore[pick, run]),
            ),
            (
                f"run{run+1} all-satisfied | available",
                allsat[full_avail, run].mean(),
                allsat[pick, run][sub_avail].mean(),
            ),
        ]
        for label, full_value, subset_value in pairs:
            report.append(f"| {label} | {full_value:.6f} | {subset_value:.6f} |")
    full_all3 = available.all(axis=1)
    subset_all3 = available[pick].all(axis=1)
    full_center_flip = (anchor_m1[full_all3] < anchor_m0[full_all3]).mean()
    subset_center_flip = (anchor_m1[pick][subset_all3] < anchor_m0[pick][subset_all3]).mean()
    report.append(f"| N=3 center flipped | {full_center_flip:.6f} | {subset_center_flip:.6f} |")
    report.append(
        f"| any per-run flip among N=3 | {flipped[full_all3].any(axis=1).mean():.6f} "
        f"| {flipped[pick][subset_all3].any(axis=1).mean():.6f} |"
    )

    report.extend(["", "### Availability pattern", "", "| bit pattern (r3r2r1) | full n (%) | subset n (%) |", "|---|---:|---:|"])
    for value in range(8):
        report.append(
            f"| {value:03b} | {full_pattern[value]} ({100*full_pattern[value]/population:.3f}%) "
            f"| {subset_pattern[value]} ({100*subset_pattern[value]/N_SUBSET:.3f}%) |"
        )

    report.extend(
        [
            "",
            "### Cross-run flip count among rows with all three draws",
            "",
            "| number of flipped draws | full n (%) | subset n (%) |",
            "|---:|---:|---:|",
        ]
    )
    full_flip_count = np.bincount(flip_count[full_all3], minlength=4)
    subset_flip_count = np.bincount(flip_count[pick][subset_all3], minlength=4)
    for value in range(4):
        report.append(
            f"| {value} | {full_flip_count[value]} ({100*full_flip_count[value]/full_all3.sum():.3f}%) "
            f"| {subset_flip_count[value]} ({100*subset_flip_count[value]/subset_all3.sum():.3f}%) |"
        )

    report.extend(
        [
            "",
            "## Constraint load and pooled difficulty",
            "",
            "Difficulty quintile 0 is the hardest and 4 is the easiest under the constraint-ID-pooled "
            "mean IFEval score across available draws.",
            "",
            "| category | level | full n (%) | subset n (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    constraint_count = frame["n_constraints"].to_numpy(int)
    for value in sorted(set(constraint_count)):
        full_count = int((constraint_count == value).sum())
        subset_count = int((constraint_count[pick] == value).sum())
        report.append(
            f"| constraints per input | {value} | {full_count} ({100*full_count/population:.3f}%) "
            f"| {subset_count} ({100*subset_count/N_SUBSET:.3f}%) |"
        )
    for value in range(5):
        full_count = int((difficulty_q == value).sum())
        subset_count = int((difficulty_q[pick] == value).sum())
        report.append(
            f"| pooled difficulty quintile | {value} | {full_count} ({100*full_count/population:.3f}%) "
            f"| {subset_count} ({100*subset_count/N_SUBSET:.3f}%) |"
        )

    report.extend(
        [
            "",
            "## Interpretable distribution quantiles",
            "",
            "Values are q05 / q25 / q50 / q75 / q95. Anchor summaries condition on the stated "
            "availability so missing values are not converted into numeric sentinels.",
            "",
            "| variable | scope | full quantiles | subset quantiles |",
            "|---|---|---:|---:|",
        ]
    )
    quantile_specs = (
        ("prompt tokens", frame["prompt_tokens"].to_numpy(float), np.ones(population, dtype=bool), "all"),
        ("constraint tokens", frame["constraint_tokens"].to_numpy(float), np.ones(population, dtype=bool), "all"),
        ("pooled IFEval score", row_ifscore, availability_count > 0, "at least one draw"),
        ("mean x-only NLL/token", anchor_m0, availability_count > 0, "at least one draw"),
        ("mean x+c NLL/token", anchor_m1, availability_count > 0, "at least one draw"),
        ("center width (m1-m0)", anchor_m1 - anchor_m0, availability_count > 0, "at least one draw"),
        ("mean x-only output tokens", mean_y0_length, availability_count > 0, "at least one draw"),
        ("mean x+c output tokens", mean_y1_length, availability_count > 0, "at least one draw"),
        ("x-only cross-draw std", m0_std, availability_count >= 2, "at least two draws"),
        ("x+c cross-draw std", m1_std, availability_count >= 2, "at least two draws"),
        ("CI95 lower endpoint", ci95_lower, full_all3, "all three draws"),
        ("CI95 upper endpoint", ci95_upper, full_all3, "all three draws"),
    )
    probabilities = [0.05, 0.25, 0.50, 0.75, 0.95]
    for label, values, mask, scope in quantile_specs:
        full_values = values[mask]
        subset_values = values[pick][mask[pick]]
        full_quantiles = " / ".join(f"{value:.4f}" for value in np.quantile(full_values, probabilities))
        subset_quantiles = " / ".join(f"{value:.4f}" for value in np.quantile(subset_values, probabilities))
        report.append(f"| {label} | {scope} | {full_quantiles} | {subset_quantiles} |")

    report.extend(
        [
            "",
            "## Largest source/domain groups",
            "",
            "| source/domain | full n (%) | subset n (%) |",
            "|---|---:|---:|",
        ]
    )
    sample_source = SOURCE[pick].sum(axis=0)
    for column in np.argsort(-full_counts["source"])[:20]:
        report.append(
            f"| {source_levels[column]} | {int(full_counts['source'][column])} "
            f"({100*full_counts['source'][column]/population:.3f}%) | {int(sample_source[column])} "
            f"({100*sample_source[column]/N_SUBSET:.3f}%) |"
        )

    report.extend(["", "## Constraint composition", "", "| instruction ID | full % | subset % |", "|---|---:|---:|"])
    sample_id = ID[pick].sum(axis=0)
    for column in np.argsort(-full_counts["id"]):
        report.append(
            f"| {all_ids[column]} | {100*full_counts['id'][column]/full_counts['id'].sum():.4f} "
            f"| {100*sample_id[column]/sample_id.sum():.4f} |"
        )

    report.extend(["", "## Canonical anchor defects", "", "| class | full n (%) | subset n (%) |", "|---|---:|---:|"])
    for value in sorted(set(defect)):
        full_count = int((defect == value).sum())
        sample_count = int((defect[pick] == value).sum())
        report.append(
            f"| {value} | {full_count} ({100*full_count/population:.3f}%) "
            f"| {sample_count} ({100*sample_count/N_SUBSET:.3f}%) |"
        )

    report.extend(["", "## Panel-level representativeness", "", "| panel | rows | max KS | TV ID | availability-pattern TV |", "|---|---:|---:|---:|---:|"])
    for panel_value, panel_name in enumerate("ABCDE"):
        panel_pick = pick[panel[pick] == panel_value]
        _, panel_ks, panel_tv, _ = score_parts(panel_pick)
        panel_pattern = np.bincount(availability_pattern[panel_pick], minlength=8)
        report.append(
            f"| {panel_name} | {len(panel_pick)} | {max(panel_ks.values()):.6f} "
            f"| {panel_tv['id']:.6f} | {tv_from_counts(panel_pattern, full_pattern):.6f} |"
        )

    report.extend(
        [
            "",
            "## Correlation replay",
            "",
            "| pair | full Spearman | subset Spearman |",
            "|---|---:|---:|",
        ]
    )
    for left, right in (
        ("anchor_m1_mean", "log_anchor_y1_tokens"),
        ("anchor_m0_mean", "log_anchor_y0_tokens"),
        ("anchor_center_width", "log_anchor_length_ratio"),
        ("difficulty", "observed_ifscore"),
    ):
        full_corr = spearmanr(clean_V[left], clean_V[right]).statistic
        subset_corr = spearmanr(clean_V[left][pick], clean_V[right][pick]).statistic
        report.append(f"| {left} ~ {right} | {full_corr:+.5f} | {subset_corr:+.5f} |")

    missing_counts = {f"run{run}": int((~available[pick, run-1]).sum()) for run in (1, 2, 3)}
    report.extend(
        [
            "",
            "## Generation work remaining",
            "",
            f"- Missing complete run1 anchors: {missing_counts['run1']:,}",
            f"- Missing complete run2 anchors: {missing_counts['run2']:,}",
            f"- Missing complete run3 anchors: {missing_counts['run3']:,}",
            f"- Rows with no currently complete draw: {int((availability_count[pick] == 0).sum()):,}",
            "",
            f"## Overall: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED — DO NOT TRAIN'}",
        ]
    )
    report_path = args.output_dir / "verification_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    provenance = {
        "version": 1,
        "seed": SEED,
        "population_rows": population,
        "subset_rows": N_SUBSET,
        "fixed_rows": N_FIXED,
        "rerandomized_draws": args.draws,
        "swap_proposals": args.swaps,
        "accepted_swaps": accepted,
        "random_composite_median": float(np.median(draw_scores)),
        "best_draw_composite": best_score,
        "polished_composite": composite,
        "checks": {
            name: {"value": value, "threshold": threshold, "pass": value < threshold}
            for name, (value, threshold) in checks.items()
        },
        "all_checks_pass": all_pass,
        "missing_selected_anchors": missing_counts,
        "manifest_sha256": sha256_file(manifest_path),
        "train_indices_sha256": sha256_file(args.output_dir / "train_indices.json"),
        "selected_audit_sha256": sha256_file(args.selected_audit),
        "seconds": time.time() - started,
    }
    with (args.output_dir / "provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
