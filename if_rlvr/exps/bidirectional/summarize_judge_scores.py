#!/usr/bin/env python
"""Summarise the LLM-verifier score distribution from a verl rollout dump.

Used to pick IF_LLM_VERIFIER_THRESHOLD before a real run: the threshold is read
once when the reward manager is constructed, so it has to be chosen up front,
and an LLM judge's score scale is model-specific.

Reads the JSONL that `trainer.rollout_data_dir` writes (one file per step,
named `<global_step>.jsonl`, one row per rollout) and reports:

  * `if_llm_verifier/score_mean` recomputed with exactly the semantics of
    verl/trainer/ppo/ray_trainer.py::collect_if_llm_verifier_metrics - the mean
    of `llm_verifier_score` over rows that were called, finite and >= 1;
  * the full 1..10 histogram, which the wandb metrics do not carry; and
  * the pass rate and mean bonus at every candidate threshold, since what
    actually matters for GRPO is that the bonus splits a prompt's group rather
    than firing for all or none of it.

Usage:
    summarize_judge_scores.py --dump-dir logs/judge_calibration/<experiment>
    summarize_judge_scores.py --dump-dir <dir> --step 1 --bonus 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys

SCORE_MIN = 1
SCORE_MAX = 10


def _resolve_files(dump_dir: str, step: int | None) -> list[str]:
    if not os.path.isdir(dump_dir):
        raise SystemExit(f"dump dir does not exist: {dump_dir}")
    entries = []
    for name in os.listdir(dump_dir):
        match = re.fullmatch(r"(\d+)\.jsonl", name)
        if match:
            entries.append((int(match.group(1)), os.path.join(dump_dir, name)))
    if not entries:
        raise SystemExit(
            f"no <step>.jsonl files under {dump_dir}. "
            "Was trainer.rollout_data_dir set, and did the step finish?"
        )
    entries.sort()
    if step is not None:
        chosen = [path for number, path in entries if number == step]
        if not chosen:
            raise SystemExit(
                f"step {step} not found in {dump_dir}; available: "
                + ", ".join(str(number) for number, _ in entries)
            )
        return chosen
    # Default to the newest step only: mixing steps would average across
    # different policies.
    return [entries[-1][1]]


def _as_float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", required=True, help="trainer.rollout_data_dir the run wrote to")
    parser.add_argument("--step", type=int, default=None, help="global step to read (default: the newest)")
    parser.add_argument("--bonus", type=float, default=0.1, help="IF_LLM_VERIFIER_BONUS, for the mean-bonus column")
    parser.add_argument(
        "--launcher",
        default="if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh",
        help="launcher named in the printed follow-up command",
    )
    args = parser.parse_args()

    paths = _resolve_files(args.dump_dir, args.step)

    total = 0
    constraint_positive = 0
    called = 0
    errored = 0
    unparseable = 0
    scores: list[int] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                if _as_float(row.get("acc"), 0.0) > 0.0:
                    constraint_positive += 1
                if _as_float(row.get("llm_verifier_called"), 0.0) <= 0.5:
                    continue
                called += 1
                if str(row.get("llm_verifier_error") or "").strip():
                    errored += 1
                score = _as_float(row.get("llm_verifier_score"))
                if math.isfinite(score) and score >= SCORE_MIN:
                    scores.append(int(round(score)))
                else:
                    # Judge answered but no score survived parsing (or the call
                    # failed): excluded from score_mean, exactly as the trainer
                    # metric excludes it, but it still gets no bonus.
                    unparseable += 1

    if not scores:
        print(f"No parseable judge scores in {', '.join(paths)}.", file=sys.stderr)
        print(
            f"rollouts={total} constraint_positive={constraint_positive} judge_called={called} "
            f"judge_errors={errored}",
            file=sys.stderr,
        )
        return 1

    score_mean = statistics.fmean(scores)
    recommended = min(SCORE_MAX, max(SCORE_MIN, int(math.floor(score_mean + 0.5))))
    ceil_alt = min(SCORE_MAX, max(SCORE_MIN, math.ceil(score_mean)))

    print(f"source                : {', '.join(paths)}")
    print(f"rollouts              : {total}")
    print(f"constraint-positive   : {constraint_positive} ({constraint_positive / total:.1%} of rollouts)")
    print(f"judge called          : {called} ({called / total:.1%} of rollouts)")
    print(f"judge errors          : {errored} ({errored / called:.2%} of calls)")
    print(f"score unparseable     : {unparseable} ({unparseable / called:.2%} of calls)")
    print()
    print(f"if_llm_verifier/score_mean : {score_mean:.4f}   (n={len(scores)})")
    print(
        f"  median={statistics.median(scores):g}  "
        f"stdev={statistics.pstdev(scores):.3f}  "
        f"min={min(scores)}  max={max(scores)}"
    )
    print()
    print("score  count    share    cum>=score")
    counts = {value: scores.count(value) for value in range(SCORE_MIN, SCORE_MAX + 1)}
    for value in range(SCORE_MIN, SCORE_MAX + 1):
        at_or_above = sum(counts[other] for other in range(value, SCORE_MAX + 1))
        print(f"{value:>5}  {counts[value]:>6}  {counts[value] / len(scores):>6.2%}  {at_or_above / len(scores):>10.2%}")
    print()
    print("candidate thresholds (pass = score >= T)")
    print("    T   pass/judged   pass/rollout   mean bonus per rollout")
    for threshold in range(SCORE_MIN, SCORE_MAX + 1):
        passing = sum(counts[value] for value in range(threshold, SCORE_MAX + 1))
        per_judged = passing / len(scores)
        per_rollout = passing / total
        marker = "  <-- recommended" if threshold == recommended else ""
        print(
            f"{threshold:>5}   {per_judged:>11.2%}   {per_rollout:>12.2%}   "
            f"{per_rollout * args.bonus:>21.4f}{marker}"
        )
    print()
    print(f"recommended threshold : {recommended}   [round(score_mean)]")
    if ceil_alt != recommended:
        print(f"  (ceil(score_mean) would be {ceil_alt})")

    passing_at_rec = sum(counts[value] for value in range(recommended, SCORE_MAX + 1)) / len(scores)
    if passing_at_rec < 0.05 or passing_at_rec > 0.95:
        print(
            f"  WARNING: {passing_at_rec:.1%} of judged rollouts pass at T={recommended}. GRPO only sees "
            "this bonus\n"
            "           where it varies WITHIN a prompt's group of n rollouts; a near-0% or near-100%\n"
            "           pass rate makes it close to a constant offset. Consider a neighbouring T."
        )

    print()
    print("Launch the real 4-epoch run with:")
    print(f"    IF_LLM_VERIFIER_THRESHOLD={recommended} bash {args.launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
