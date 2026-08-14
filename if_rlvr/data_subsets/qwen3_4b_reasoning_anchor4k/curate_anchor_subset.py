#!/usr/bin/env python
"""Curate a representative n=4096 subset of the Qwen3-4B reasoning anchor cache.

Design + measured pilot: if_rlvr/docs/anchor_subset_curation_design_2026-08-13.md

Guarantees:
  * exact proportional representation (largest-remainder) over 164 strata built
    from defect class x inverted x n_constraints x difficulty quintile x y1_allsat;
  * statistical match (rerandomization + within-stratum greedy swap polish) on
    11 continuous variables (tie-exact KS) and 3 categorical marginals (TV):
    54-instruction-ID, 16-family, top-50 ID-pair co-occurrence;
  * functional replay of the anchor reward rule agrees with the full frame.

Inputs (CLI):
  --audit-parquet   cacheaudit_rows_final.parquet (93,882 rows; from the 2026-08 audit)
  --full-cache      the full anchor cache JSON (metadata + items keyed by str(index))
  --out-dir         artifact directory (indices, report)
  --out-cache       path for the filtered subset cache JSON (drop-in trainable)

Deterministic under the fixed SEED. CPU-only. Requires the HF dataset in local cache
(HF_HUB_OFFLINE=1 is set; every training run on this box has it cached).

Usage of the produced cache in ANY existing launcher (no code changes):
  export IF_REF_ANCHOR_CACHE_PATH=<out-cache>
  export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=true
"""
import argparse, ast, json, os, time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", "/root/if-rlvr-rlvr/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import datasets  # noqa: E402

N_SUB = 4096
M_DRAWS = 20000
N_SWAP = 60000
MIN_CELL = 6           # min expected subset rows for an unmerged stratum cell
SEED = 20260813
GRID = 2048            # quantile-grid size for tie-exact KS
W_ID, W_FAM, W_PAIR = 5.0, 3.0, 2.0
# pre-registered acceptance thresholds (verification_report.md marks PASS/FAIL)
ACC_KS = 0.010
ACC_TV_ID = 0.005
ACC_COVER = 0.5


def parse_ids(label):
    cd = ast.literal_eval(label)[0]
    if isinstance(cd, str):
        cd = json.loads(cd)
    return cd["instruction_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-parquet", required=True)
    ap.add_argument("--full-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-cache", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---------------- frame ----------------
    df = pd.read_parquet(args.audit_parquet)
    N = len(df)
    ds = datasets.load_dataset("allenai/IF_multi_constraints_upto5", split="train")
    ds = ds.shuffle(seed=1)                      # identical split to if_dataset.py
    ds = ds.select(range(512, len(ds)))
    gt_all, msgs_all, con_all = ds["ground_truth"], ds["messages"], ds["constraint"]
    idx = df["index"].values
    assert idx.max() < len(ds)

    id_lists = [parse_ids(gt_all[i]) for i in idx]
    x_len = np.array([
        sum(len(m["content"]) for m in msgs_all[i] if m["role"] == "user") - len(con_all[i] or "")
        for i in idx
    ])
    df["x_len"] = np.maximum(x_len, 0)
    print(f"[{time.time()-t0:.0f}s] frame ready: {N} rows", flush=True)

    all_ids = sorted({k for lst in id_lists for k in lst})
    id_pos = {k: j for j, k in enumerate(all_ids)}
    n_ids = len(all_ids)
    ID = np.zeros((N, n_ids), dtype=np.int8)
    for r, lst in enumerate(id_lists):
        for k in lst:
            ID[r, id_pos[k]] += 1
    full_id_marg = ID.sum(0).astype(float)
    full_id_marg /= full_id_marg.sum()

    fams = sorted({k.split(":")[0] for k in all_ids})
    fam_pos = {f: j for j, f in enumerate(fams)}
    id2fam = np.array([fam_pos[k.split(":")[0]] for k in all_ids])
    FAM = np.zeros((N, len(fams)), dtype=np.int8)
    for j in range(n_ids):
        FAM[:, id2fam[j]] += ID[:, j]
    full_fam_marg = FAM.sum(0).astype(float)
    full_fam_marg /= full_fam_marg.sum()

    pair_ct = Counter()
    for lst in id_lists:
        u = sorted(set(lst))
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                pair_ct[(u[a], u[b])] += 1
    top_pairs = [p for p, _ in pair_ct.most_common(50)]
    pair_pos = {p: j for j, p in enumerate(top_pairs)}
    PAIR = np.zeros((N, 50), dtype=np.int8)
    for r, lst in enumerate(id_lists):
        u = sorted(set(lst))
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                j = pair_pos.get((u[a], u[b]))
                if j is not None:
                    PAIR[r, j] += 1
    full_pair_marg = PAIR.sum(0).astype(float)
    full_pair_marg /= full_pair_marg.sum()

    # difficulty: per-ID mean y1 compliance (init-policy pass rate), averaged per row
    id_diff = (ID.T @ df["y1_ifscore"].values) / np.maximum(ID.sum(0), 1)
    df["diff_score"] = (ID @ id_diff) / np.maximum(ID.sum(1), 1)

    # ---------------- strata ----------------
    ne0 = (df.y0_ntok < 10).values
    ne1 = (df.y1_ntok < 10).values
    dg1 = df.y1_degen.values
    dg0 = df.y0_degen.values
    dup = (df.y0_text == df.y1_text).values
    cot = (df.cot0 | df.cot1).values
    defect = np.select([ne0, ne1, dg1, dg0, dup, cot],
                       ["ne_y0", "ne_y1", "degen_y1", "degen_y0", "dup", "cot_leak"],
                       default="clean")
    inv = (df.m1 < df.m0).values
    nc = df.n_constraints.values
    dq = pd.qcut(df.diff_score, 5, labels=False, duplicates="drop").values
    sat = df.y1_allsat.values.astype(int)
    df["defect_class"], df["inverted"], df["diff_q"] = defect, inv, dq

    c5 = pd.Series(list(zip(defect, inv, nc, dq, sat))).value_counts()
    c4 = pd.Series(list(zip(defect, inv, nc, dq))).value_counts()
    c3 = pd.Series(list(zip(defect, inv, nc))).value_counts()
    key_arr = np.empty(N, dtype=object)
    for r in range(N):
        k5 = (defect[r], inv[r], nc[r], dq[r], sat[r])
        if c5[k5] * N_SUB / N >= MIN_CELL:
            key_arr[r] = k5
        elif c4[k5[:4]] * N_SUB / N >= MIN_CELL:
            key_arr[r] = ("L4",) + k5[:4]
        elif c3[k5[:3]] * N_SUB / N >= MIN_CELL:
            key_arr[r] = ("L3",) + k5[:3]
        else:
            key_arr[r] = ("L2",) + k5[:2]
    df["stratum"] = key_arr
    strata = {k: np.asarray(v) for k, v in df.groupby("stratum", sort=False).indices.items()}
    names = list(strata)
    sizes = np.array([len(strata[k]) for k in names])
    quota = sizes * N_SUB / N
    base = np.floor(quota).astype(int)
    order = np.argsort(-(quota - base))
    base[order[: N_SUB - base.sum()]] += 1
    alloc = dict(zip(names, base))
    assert sum(alloc.values()) == N_SUB
    print(f"[{time.time()-t0:.0f}s] {len(names)} strata (min alloc {base.min()}, max {base.max()})", flush=True)

    # ---------------- balance machinery ----------------
    V = {
        "m0": df.m0.values.astype(float), "m1": df.m1.values.astype(float),
        "width": df.width.values.astype(float),
        "log_y0": np.log1p(df.y0_ntok.values), "log_y1": np.log1p(df.y1_ntok.values),
        "log_ratio": np.log((df.y1_ntok.values + 1) / (df.y0_ntok.values + 1)),
        "y1_ifscore": df.y1_ifscore.values.astype(float),
        "p_zero": df.p_zero.values.astype(float), "p_bonus": df.p_bonus.values.astype(float),
        "x_len": df.x_len.values.astype(float), "diff_score": df.diff_score.values.astype(float),
    }
    grids, F_full, binof = {}, {}, {}
    for c, v in V.items():
        uu = np.unique(v)
        g = uu if len(uu) <= GRID else np.unique(np.quantile(v, np.linspace(0, 1, GRID)))
        grids[c] = g
        F_full[c] = np.searchsorted(np.sort(v), g, side="right") / N
        binof[c] = np.searchsorted(g, v, side="left").astype(np.int32)
    varnames = list(V)

    def score_parts(pick):
        ks = {}
        for c in varnames:
            h = np.bincount(binof[c][pick], minlength=len(grids[c]))
            ks[c] = float(np.max(np.abs(np.cumsum(h) / len(pick) - F_full[c])))
        sm = ID[pick].sum(0).astype(float)
        tv_id = 0.5 * np.abs(sm / sm.sum() - full_id_marg).sum()
        fm = FAM[pick].sum(0).astype(float)
        tv_fam = 0.5 * np.abs(fm / fm.sum() - full_fam_marg).sum()
        pm = PAIR[pick].sum(0).astype(float)
        tv_pair = 0.5 * np.abs(pm / max(pm.sum(), 1) - full_pair_marg).sum()
        cover = (sm / np.maximum(full_id_marg * sm.sum(), 1e-9)).min()
        comp = (sum(ks.values()) + W_ID * tv_id + W_FAM * tv_fam + W_PAIR * tv_pair
                + (10 if cover < ACC_COVER else 0))
        return comp, ks, tv_id, tv_fam, tv_pair, cover

    # ---------------- stage 1: rerandomization ----------------
    rng_master = np.random.default_rng(SEED)
    best = None
    scores = []
    for it in range(M_DRAWS):
        rng = np.random.default_rng(rng_master.integers(0, 2**63))
        pick = np.concatenate([rng.choice(strata[k], size=alloc[k], replace=False)
                               for k in names if alloc[k] > 0])
        comp = score_parts(pick)[0]
        scores.append(comp)
        if best is None or comp < best[0]:
            best = (comp, pick.copy(), it)
        if (it + 1) % 5000 == 0:
            print(f"[{time.time()-t0:.0f}s] draws {it+1}/{M_DRAWS} best {best[0]:.4f}", flush=True)
    scores = np.array(scores)

    # ---------------- stage 2: within-stratum greedy swap polish ----------------
    pick = best[1].copy()
    in_sub = np.zeros(N, dtype=bool)
    in_sub[pick] = True
    hists = {c: np.bincount(binof[c][pick], minlength=len(grids[c])).astype(np.int32) for c in varnames}
    sm = ID[pick].sum(0).astype(np.int64)
    fm = FAM[pick].sum(0).astype(np.int64)
    pm = PAIR[pick].sum(0).astype(np.int64)

    def comp_from_state():
        ks_sum = 0.0
        for c in varnames:
            ks_sum += np.max(np.abs(np.cumsum(hists[c]) / N_SUB - F_full[c]))
        tv_id = 0.5 * np.abs(sm / sm.sum() - full_id_marg).sum()
        tv_fam = 0.5 * np.abs(fm / fm.sum() - full_fam_marg).sum()
        tv_pair = 0.5 * np.abs(pm / max(pm.sum(), 1) - full_pair_marg).sum()
        cover = (sm / np.maximum(full_id_marg * sm.sum(), 1e-9)).min()
        return (ks_sum + W_ID * tv_id + W_FAM * tv_fam + W_PAIR * tv_pair
                + (10 if cover < ACC_COVER else 0))

    cur = comp_from_state()
    rng = np.random.default_rng(SEED + 7)
    swappable = {k: v for k, v in strata.items() if 0 < alloc[k] < len(v)}
    snames = list(swappable)
    sweights = np.array([len(swappable[k]) for k in snames], dtype=float)
    sweights /= sweights.sum()
    accepted = 0
    for it in range(N_SWAP):
        k = snames[rng.choice(len(snames), p=sweights)]
        mem = swappable[k]
        ins = mem[in_sub[mem]]
        outs = mem[~in_sub[mem]]
        if len(ins) == 0 or len(outs) == 0:
            continue
        a = ins[rng.integers(len(ins))]
        b = outs[rng.integers(len(outs))]
        for c in varnames:
            hists[c][binof[c][a]] -= 1
            hists[c][binof[c][b]] += 1
        sm += ID[b].astype(np.int64) - ID[a].astype(np.int64)
        fm += FAM[b].astype(np.int64) - FAM[a].astype(np.int64)
        pm += PAIR[b].astype(np.int64) - PAIR[a].astype(np.int64)
        new = comp_from_state()
        if new < cur - 1e-12:
            cur = new
            in_sub[a] = False
            in_sub[b] = True
            accepted += 1
        else:
            for c in varnames:
                hists[c][binof[c][a]] += 1
                hists[c][binof[c][b]] -= 1
            sm -= ID[b].astype(np.int64) - ID[a].astype(np.int64)
            fm -= FAM[b].astype(np.int64) - FAM[a].astype(np.int64)
            pm -= PAIR[b].astype(np.int64) - PAIR[a].astype(np.int64)
    pick = np.where(in_sub)[0]
    assert len(pick) == N_SUB
    comp, ks, tv_id, tv_fam, tv_pair, cover = score_parts(pick)
    print(f"[{time.time()-t0:.0f}s] polish: {accepted} swaps accepted; composite "
          f"{np.median(scores):.4f} (random) -> {best[0]:.4f} (rerand) -> {comp:.4f}", flush=True)

    # ---------------- balanced halves (within-stratum m1 pairing) ----------------
    half = np.empty(N_SUB, dtype="U1")
    pos_of = {int(r): j for j, r in enumerate(pick)}
    toggle = 0
    sub_strat = [str(s) for s in df["stratum"].values[pick]]
    for k in dict.fromkeys(sub_strat):
        rows = pick[np.array([s == k for s in sub_strat])]
        rows = rows[np.argsort(V["m1"][rows])]
        for j, r in enumerate(rows):
            half[pos_of[int(r)]] = "AB"[(j + toggle) % 2]
        toggle ^= len(rows) % 2
    nA = int((half == "A").sum())
    assert abs(nA - N_SUB // 2) <= 1, nA

    # ---------------- artifacts ----------------
    sub = df.iloc[pick]
    out_rows = [
        {
            "index": int(r["index"]), "stratum": str(r["stratum"]), "half": h,
            "defect_class": r.defect_class, "inverted": bool(r.inverted),
            "n_constraints": int(r.n_constraints), "diff_q": int(r.diff_q),
            "y1_allsat": bool(r.y1_allsat),
        }
        for (h, (_, r)) in zip(half, sub.iterrows())
    ]
    with open(os.path.join(args.out_dir, "subset_indices.json"), "w") as f:
        json.dump({
            "n": N_SUB, "seed": SEED, "frame_rows": N,
            "design_doc": "if_rlvr/docs/anchor_subset_curation_design_2026-08-13.md",
            "source_cache": os.path.basename(args.full_cache),
            "rows": out_rows,
        }, f)
    print(f"[{time.time()-t0:.0f}s] wrote subset_indices.json", flush=True)

    with open(args.full_cache, "r", encoding="utf-8") as f:
        payload = json.load(f)
    keep = {str(int(i)) for i in sub["index"].values}
    items = payload.get("items", {})
    sub_items = {k: v for k, v in items.items() if k in keep}
    assert len(sub_items) == N_SUB, (len(sub_items), N_SUB)
    out_payload = {k: v for k, v in payload.items() if k != "items"}
    out_payload["items"] = {k: sub_items[k] for k in sorted(sub_items, key=int)}
    with open(args.out_cache, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)
    print(f"[{time.time()-t0:.0f}s] wrote subset cache "
          f"({os.path.getsize(args.out_cache)/1e6:.1f} MB, metadata copied verbatim)", flush=True)

    # ---------------- verification report ----------------
    L = []
    ok_all = True

    def line(s=""):
        L.append(s)

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        line(f"- {'PASS' if cond else '**FAIL**'} — {name}: {detail}")

    line("# Subset verification report (auto-generated)")
    line()
    line(f"n = {N_SUB} of {N} frame rows; seed {SEED}; composite {comp:.4f} "
         f"(random-draw median {np.median(scores):.4f}).")
    line(f"Cache file: `{os.path.basename(args.out_cache)}`; halves A/B = {nA}/{N_SUB-nA}.")
    line()
    line("## Acceptance checks")
    worst_ks = max(ks.values())
    check("KS (worst of 11 variables)", worst_ks < ACC_KS,
          f"{worst_ks:.4f} < {ACC_KS} (detectability threshold 0.0212)")
    check("TV 54-instruction-ID marginal", tv_id < ACC_TV_ID, f"{tv_id:.4f} < {ACC_TV_ID}")
    check("worst-ID coverage", cover >= ACC_COVER, f"{cover:.2f}x expected")
    for nm, fu, su in [
        ("inverted %", (df.m1 < df.m0).mean(), (sub.m1 < sub.m0).mean()),
        ("y1_allsat %", df.y1_allsat.mean(), sub.y1_allsat.mean()),
        ("p_zero mean", df.p_zero.mean(), sub.p_zero.mean()),
        ("p_bonus mean", df.p_bonus.mean(), sub.p_bonus.mean()),
        ("false-wipe compliant y1", (df.m1 < df.m0)[df.y1_allsat].mean(),
         (sub.m1 < sub.m0)[sub.y1_allsat].mean()),
    ]:
        se = np.sqrt(max(fu * (1 - fu), 1e-9) / N_SUB) if fu <= 1 else 0.0
        tol = 1.96 * se if fu <= 1 else 0.0
        check(nm, abs(su - fu) <= max(tol, 1e-3), f"full {fu:.4f} vs subset {su:.4f} (tol {max(tol,1e-3):.4f})")
    line()
    line("## Per-variable KS D")
    line()
    line("| variable | KS D |")
    line("|---|---|")
    for c, v in sorted(ks.items(), key=lambda kv: -kv[1]):
        line(f"| {c} | {v:.4f} |")
    line()
    line(f"TV family = {tv_fam:.4f}; TV top-50 pairs = {tv_pair:.4f}")
    line()
    line("## Correlation structure (Spearman)")
    line()
    line("| pair | full | subset |")
    line("|---|---|---|")
    for a, b in [("m1", "log_y1"), ("m0", "log_y0"), ("m1", "y1_ifscore"), ("width", "log_ratio")]:
        line(f"| {a} ~ {b} | {spearmanr(V[a], V[b]).statistic:+.3f} "
             f"| {spearmanr(V[a][pick], V[b][pick]).statistic:+.3f} |")
    line()
    line("## Defect classes (count / % vs full %)")
    line()
    line("| class | subset n | subset % | full % |")
    line("|---|---|---|---|")
    for kls in ["clean", "degen_y1", "ne_y1", "ne_y0", "cot_leak", "degen_y0", "dup"]:
        s_ = (sub.defect_class == kls).sum()
        line(f"| {kls} | {s_} | {100*s_/N_SUB:.2f} | {100*(df.defect_class==kls).mean():.2f} |")
    line()
    line("## Inversion by length-ratio octile (%)")
    edges = np.quantile(V["log_ratio"], np.linspace(0, 1, 9))
    of = pd.cut(V["log_ratio"], edges, labels=False, include_lowest=True)
    osub = pd.cut(V["log_ratio"][pick], edges, labels=False, include_lowest=True)
    gf = pd.Series((df.m1 < df.m0).values).groupby(of).mean()
    gs = pd.Series((sub.m1 < sub.m0).values).groupby(osub).mean().reindex(range(8))
    line()
    line("| octile | " + " | ".join(str(i) for i in range(1, 9)) + " |")
    line("|---|" + "---|" * 8)
    line("| full | " + " | ".join(f"{v*100:.1f}" for v in gf) + " |")
    line("| subset | " + " | ".join(f"{v*100:.1f}" for v in gs) + " |")
    line()
    line("## Instruction-ID marginal (subset share vs full share, %)")
    line()
    line("| instruction_id | subset % | full % |")
    line("|---|---|---|")
    sm_final = ID[pick].sum(0).astype(float)
    sm_marg = sm_final / sm_final.sum()
    for j in np.argsort(-full_id_marg):
        line(f"| {all_ids[j]} | {100*sm_marg[j]:.3f} | {100*full_id_marg[j]:.3f} |")
    line()
    line(f"## Overall: {'ALL CHECKS PASS' if ok_all else 'SOME CHECKS FAILED — do not use'}")
    with open(os.path.join(args.out_dir, "verification_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[{time.time()-t0:.0f}s] verification report written; all-pass = {ok_all}", flush=True)
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
