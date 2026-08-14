#!/usr/bin/env python
"""Compute per-row entropy/varentropy sums for the subset anchors (z-rule prerequisite).

For the typicality z-score z(t) = (sum NLL_i - sum H_i) / sqrt(sum V_i) we need,
for every cached y0/y1, the reference model's predictive entropy H_i and
varentropy V_i at each scored position, conditioned on the SAME x-only prefix
the pipeline scores with (enable_thinking=False chat template). This script:

  1. rebuilds the x prefix exactly like the training pipeline
     (if_dataset._make_constraint_free_messages + apply_chat_template with
     add_generation_prompt=True, enable_thinking=False);
  2. runs an HF forward pass (full logits; vLLM prompt_logprobs cannot give
     full-vocab entropy) and reduces, per position, in fp32:
       H = -sum p log p,  V = sum p (log p)^2 - H^2
     under BOTH the full distribution and the sampling-truncated one
     (top_k=20 then top_p=0.95, renormalized — the regime anchors were drawn from);
  3. SELF-VALIDATES by recomputing sum NLL of the cached tokens and comparing
     to the cached ref*_nll (proves the prefix reconstruction is identical);
  4. writes a sidecar JSON next to the cache: <cache>.hv.json

Shard across GPUs: run N copies with --shard i --num-shards N (each writes
<cache>.hv.shard{i}.json); then --merge to combine.
"""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "if_rlvr"))
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".cache", "huggingface"))

DEFAULT_CACHE = os.path.join(
    REPO, ".cache",
    "if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json")


def merge(cache_path):
    base = cache_path + ".hv"
    out = {}
    i = 0
    while os.path.exists(f"{base}.shard{i}.json"):
        out.update(json.load(open(f"{base}.shard{i}.json"))["rows"])
        i += 1
    if not out:
        raise SystemExit("no shard files found")
    json.dump({"rows": out, "n": len(out)}, open(base + ".json", "w"))
    print(f"merged {i} shards -> {base}.json ({len(out)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--merge", action="store_true", help="merge shard outputs and exit")
    args = ap.parse_args()
    if args.merge:
        merge(args.cache)
        return

    import torch
    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from if_dataset import _make_constraint_free_messages

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    payload = json.load(open(args.cache))
    items = {int(k): v for k, v in payload["items"].items()}
    idx_all = sorted(items)
    idx = [i for j, i in enumerate(idx_all) if j % args.num_shards == args.shard]

    ds = datasets.load_dataset("allenai/IF_multi_constraints_upto5", split="train")
    ds = ds.shuffle(seed=1)
    ds = ds.select(range(512, len(ds)))

    def reductions(logits_f32):
        """per-position H, V (full) and H, V (top-k -> top-p truncated, renormalized)."""
        logp = torch.log_softmax(logits_f32, dim=-1)
        p = logp.exp()
        H = -(p * logp).sum(-1)
        V = (p * logp.pow(2)).sum(-1) - H.pow(2)
        pk, _ = p.topk(args.top_k, dim=-1)                     # [T, k] descending
        pk = pk / pk.sum(-1, keepdim=True)
        cum = pk.cumsum(-1)
        keep = (cum - pk) < args.top_p                          # minimal set with cum >= top_p
        pt = torch.where(keep, pk, torch.zeros_like(pk))
        pt = pt / pt.sum(-1, keepdim=True)
        lt = torch.where(pt > 0, pt.log(), torch.zeros_like(pt))
        Ht = -(pt * lt).sum(-1)
        Vt = (pt * lt.pow(2)).sum(-1) - Ht.pow(2)
        return H, V, Ht, Vt

    rows_out = {}
    t0 = time.time()
    nll_diffs = []
    with torch.no_grad():
        for n_done, i in enumerate(idx):
            it = items[i]
            x_msgs = _make_constraint_free_messages(ds[int(i)])
            prefix = tok.apply_chat_template(x_msgs, add_generation_prompt=True,
                                             enable_thinking=False, tokenize=True)
            row = {}
            for tag, key in (("0", "y0"), ("1", "y1")):
                y = list(it[key])
                seq = torch.tensor([prefix + y], device="cuda")
                out = model(seq).logits[0]                       # [L, vocab]
                sl = slice(len(prefix) - 1, len(prefix) + len(y) - 1)
                sH = sV = sHt = sVt = snll = 0.0
                targets = torch.tensor(y, device="cuda")
                pos_logits = out[sl]
                for s in range(0, pos_logits.shape[0], 2048):
                    chunk = pos_logits[s:s + 2048].float()
                    H, V, Ht, Vt = reductions(chunk)
                    sH += float(H.sum()); sV += float(V.sum())
                    sHt += float(Ht.sum()); sVt += float(Vt.sum())
                    lp = torch.log_softmax(chunk, dim=-1)
                    snll += float(-lp.gather(-1, targets[s:s + 2048, None]).sum())
                row[f"sum_H{tag}_full"] = sH
                row[f"sum_V{tag}_full"] = sV
                row[f"sum_H{tag}_trunc"] = sHt
                row[f"sum_V{tag}_trunc"] = sVt
                row[f"nll{tag}_recomputed"] = snll
                cached = float(it[f"ref{tag}_nll"])
                nll_diffs.append(abs(snll - cached) / max(len(y), 1))
                del out, pos_logits
            rows_out[str(i)] = row
            if (n_done + 1) % 100 == 0:
                md = sorted(nll_diffs)
                print(f"[{time.time()-t0:6.0f}s] {n_done+1}/{len(idx)} rows | "
                      f"per-token |NLL diff| p50 {md[len(md)//2]:.4f} p99 {md[int(len(md)*0.99)]:.4f}", flush=True)

    md = sorted(nll_diffs)
    summary = {
        "n_rows": len(rows_out),
        "per_token_nll_absdiff_p50": md[len(md) // 2],
        "per_token_nll_absdiff_p99": md[int(len(md) * 0.99)],
        "note": "diff vs cached vLLM scoring; large values (>0.05/token) mean prefix mismatch — investigate before using H/V",
        "truncation": {"top_k": args.top_k, "top_p": args.top_p},
    }
    out_path = f"{args.cache}.hv.shard{args.shard}.json"
    json.dump({"rows": rows_out, "summary": summary}, open(out_path, "w"))
    print(f"wrote {out_path}")
    print(f"validation: per-token |NLL diff| p50 {summary['per_token_nll_absdiff_p50']:.4f}, "
          f"p99 {summary['per_token_nll_absdiff_p99']:.4f}")


if __name__ == "__main__":
    main()
