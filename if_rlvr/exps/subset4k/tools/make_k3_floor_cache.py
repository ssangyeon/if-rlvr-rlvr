#!/usr/bin/env python
"""Build the k=3 loosened-floor cache for arm b2 from three independent y0 draws.

Inputs: the final SUBSET4096 cache (--base; provides y1/B, metadata, and the
draw-1 fallback) plus two or three additional anchor caches whose rows carry
independent y0 draws for the same indices (--draws, each produced by re-running
the v2 precompute to a fresh cache path).

The derived cache replaces each row's ref0 statistics so that the trainer's
ref0_mean equals the chosen k-draw floor form. No trainer changes: the trainer
keeps computing ref0_nll / ref0_token_count. The stored y0 token ids remain
draw-1's text (provenance), and ref0_ppl is kept consistent (exp of the mean).

Forms (per row, over the k available m0 draws):
  min3      A' = min(draws)                — distribution-free; never tighter than
                                             the loosest draw; P(new draw < min) = 1/(k+1)
  mean_zs95 A' = mean - 2.92*s*sqrt(1+1/k) — one-sided 95% prediction bound (t, df=k-1)
  mean_zs99 A' = mean - 6.96*s*sqrt(1+1/k) — one-sided 99% prediction bound
CAUTION (measured motivation for min3): with k=3 the sample std has 2 degrees of
freedom; when draws agree by luck, mean-z*s floors land TIGHTER than a single
draw. The report quantifies this per form before you pick one.

Rows missing a complete y0 in some draw fall back to the base single draw and
are counted in the report.
"""
import argparse, json, math, os
import statistics

T95 = {2: 2.920, 3: 2.353}  # one-sided t, df = k-1
T99 = {2: 6.965, 3: 4.541}


def complete0(it):
    return bool(it.get("y0")) and int(it.get("ref0_token_count", 0) or 0) > 0 \
        and math.isfinite(float(it.get("ref0_nll", float("inf"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="final SUBSET4096 cache json")
    ap.add_argument("--draws", nargs="+", required=True,
                    help="1-2 additional anchor caches with independent y0 draws (base counts as draw 1)")
    ap.add_argument("--form", choices=["min3", "mean_zs95", "mean_zs99"], default="min3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base))
    items = base["items"]
    draw_maps = []
    for p in args.draws:
        d = json.load(open(p))["items"]
        draw_maps.append({k: it for k, it in d.items() if complete0(it)})

    n_fallback = 0
    tighter_than_single = 0
    per_row_std = []
    deltas = []
    out_items = {}
    for k, it in items.items():
        m0s = [float(it["ref0_nll"]) / int(it["ref0_token_count"])]
        for dm in draw_maps:
            if k in dm:
                d = dm[k]
                m0s.append(float(d["ref0_nll"]) / int(d["ref0_token_count"]))
        new = dict(it)
        if len(m0s) < 2:
            n_fallback += 1
            out_items[k] = new
            continue
        kk = len(m0s)
        mean = sum(m0s) / kk
        s = statistics.stdev(m0s)
        per_row_std.append(s)
        if args.form == "min3":
            a_prime = min(m0s)
        elif args.form == "mean_zs95":
            a_prime = mean - T95[kk - 1] * s * math.sqrt(1 + 1 / kk)
        else:
            a_prime = mean - T99[kk - 1] * s * math.sqrt(1 + 1 / kk)
        if a_prime > m0s[0]:
            tighter_than_single += 1
        deltas.append(m0s[0] - a_prime)
        cnt = int(it["ref0_token_count"])
        new["ref0_nll"] = a_prime * cnt
        new["ref0_ppl"] = math.exp(a_prime)
        out_items[k] = new

    out = {"metadata": base["metadata"], "items": out_items}
    json.dump(out, open(args.out, "w"))
    n = len(items)
    print(f"wrote {args.out}")
    print(f"rows: {n}; k>=2 available: {n - n_fallback}; single-draw fallback: {n_fallback}")
    if per_row_std:
        ps = sorted(per_row_std)
        print(f"per-row anchor std (m0, nats): p50 {ps[len(ps)//2]:.3f}  p90 {ps[int(len(ps)*0.9)]:.3f}")
        ds = sorted(deltas)
        print(f"floor loosening (A - A', nats): p50 {ds[len(ds)//2]:.3f}  p90 {ds[int(len(ds)*0.9)]:.3f}")
        print(f"rows where {args.form} is TIGHTER than the single draw: {tighter_than_single} "
              f"({100*tighter_than_single/max(len(deltas),1):.2f}%)"
              + ("  <-- should be 0 for min3" if args.form == "min3" else ""))
    # pseudo-wipe replay under the new floor (y1 as init rollout), c = 0
    wipes = 0
    for k, it in out_items.items():
        m0 = float(it["ref0_nll"]) / int(it["ref0_token_count"])
        m1 = float(it["ref1_nll"]) / int(it["ref1_token_count"])
        wipes += int(m1 < m0)
    print(f"pseudo-wipe (y1 replay) under {args.form} floor: {100*wipes/n:.2f}%  "
          f"(compare against the base single-draw rate and the b1 margin curve)")


if __name__ == "__main__":
    main()
