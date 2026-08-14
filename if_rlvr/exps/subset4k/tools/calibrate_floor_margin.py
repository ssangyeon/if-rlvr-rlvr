#!/usr/bin/env python
"""Calibrate the Stage-2 floor margin c (arm b1) from the subset cache, offline.

Pseudo-rollout replay: at initialization the policy IS the reference, so the
reference's own y1 (an answer to x+c) is the best zero-cost stand-in for an
init rollout. For a grid of margins c, we measure the pseudo-wipe rate
P(m1 < m0 - c) over the 4,096 rows and pick the smallest c whose rate is at or
under the pre-registered target (default 2%).

Robustness view: the same curve restricted to rows whose v1 y1 passed all its
own constraints (compliance flags carried in subset_indices.json) — the
"wiping a correct answer" analog of the historical 11.5% false-wipe number.

Writes if_rlvr/exps/subset4k/calibration_floor_margin.json, which
b1_floor_margin.sh reads for its default IF_PPL_ANCHOR_FLOOR_MARGIN.
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        REPO, ".cache",
        "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json"))
    ap.add_argument("--subset-indices", default=os.path.join(
        REPO, "if_rlvr", "data_subsets", "qwen3_4b_reasoning_anchor4k", "subset_indices.json"))
    ap.add_argument("--target-wipe", type=float, default=0.02)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "calibration_floor_margin.json"))
    args = ap.parse_args()

    payload = json.load(open(args.cache))
    items = payload["items"]
    sub = json.load(open(args.subset_indices))
    allsat_v1 = {int(r["index"]): bool(r["y1_allsat"]) for r in sub["rows"]}

    rows = []
    for k, it in items.items():
        m0 = float(it["ref0_nll"]) / int(it["ref0_token_count"])
        m1 = float(it["ref1_nll"]) / int(it["ref1_token_count"])
        if math.isfinite(m0) and math.isfinite(m1):
            rows.append((int(k), m0, m1))
    n = len(rows)
    comp = [(i, m0, m1) for i, m0, m1 in rows if allsat_v1.get(i, False)]

    grid = [round(c * 0.05, 2) for c in range(0, 41)]  # 0.00 .. 2.00 nats
    table = []
    rec = None
    for c in grid:
        wipe = sum(1 for _, m0, m1 in rows if m1 < m0 - c) / n
        wipe_comp = (sum(1 for _, m0, m1 in comp if m1 < m0 - c) / len(comp)) if comp else float("nan")
        table.append({"c": c, "pseudo_wipe": round(wipe, 4), "pseudo_wipe_v1compliant": round(wipe_comp, 4)})
        if rec is None and wipe <= args.target_wipe:
            rec = c
    if rec is None:
        rec = grid[-1]

    print(f"rows={n} (v1-compliant subset: {len(comp)}), target pseudo-wipe <= {args.target_wipe:.0%}")
    print(f"{'c':>6} {'wipe':>8} {'wipe|compliant':>15}")
    for t in table:
        mark = "  <-- recommended" if t["c"] == rec else ""
        if t["c"] in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0) or t["c"] == rec:
            print(f"{t['c']:>6} {t['pseudo_wipe']:>8.2%} {t['pseudo_wipe_v1compliant']:>14.2%}{mark}")

    out = {
        "recommended_c": rec,
        "target_wipe": args.target_wipe,
        "method": "pseudo-rollout replay: y1 as init rollout; P(m1 < m0 - c) over the subset",
        "n_rows": n,
        "curve": table,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {os.path.abspath(args.out)} (recommended_c={rec})")


if __name__ == "__main__":
    main()
