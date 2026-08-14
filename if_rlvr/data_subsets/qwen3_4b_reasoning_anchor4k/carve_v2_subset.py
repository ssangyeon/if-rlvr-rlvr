#!/usr/bin/env python
"""Carve the v2 subset cache (4,096 rows) from the completed full v2 anchor cache.

Same procedure as the v1 carve: filter items to the subset indices, copy the
metadata block verbatim. Refuses to write if any subset index is incomplete in
the source file (prints the missing list — feed it to the regeneration
work-order, or run the stratum-repair instead).

Usage:
  python carve_v2_subset.py --full-cache <completed v2 cache json> \
      --out .cache/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))


def complete(it) -> bool:
    y0, y1 = it.get("y0") or [], it.get("y1") or []
    return (
        bool(y0) and bool(y1)
        and int(it.get("ref0_token_count", 0) or 0) > 0
        and int(it.get("ref1_token_count", 0) or 0) > 0
        and math.isfinite(float(it.get("ref0_nll", float("inf"))))
        and math.isfinite(float(it.get("ref1_nll", float("inf"))))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-cache", required=True)
    ap.add_argument("--subset-indices", default=os.path.join(HERE, "subset_indices.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sub = json.load(open(args.subset_indices))
    idx = [int(r["index"]) for r in sub["rows"]]
    assert len(idx) == len(set(idx)) == 4096, len(idx)

    payload = json.load(open(args.full_cache))
    items = payload.get("items", {})
    missing = [i for i in idx if str(i) not in items or not complete(items[str(i)])]
    if missing:
        out_missing = os.path.join(HERE, "v2_missing_subset_indices.latest.json")
        json.dump({"n_missing": len(missing), "indices": sorted(missing)}, open(out_missing, "w"))
        raise SystemExit(
            f"REFUSING to carve: {len(missing)} subset rows incomplete in {args.full_cache}.\n"
            f"Missing index list written to {out_missing} — regenerate them (work-order flow) "
            f"or run the stratum-repair before carving."
        )

    out_payload = {k: v for k, v in payload.items() if k != "items"}
    out_payload["items"] = {str(i): items[str(i)] for i in sorted(idx)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, 4096 items, metadata verbatim)")


if __name__ == "__main__":
    main()
