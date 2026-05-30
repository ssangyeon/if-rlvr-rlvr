#!/usr/bin/env python
"""Convert ``allenai/IF_multi_constraints_upto5`` to verl RLVR parquet.

This is the *exact* training data used by open-instruct's
``scripts/train/rlvr/valpy_if_grpo_fast.sh`` (``--dataset_mixer_list
allenai/IF_multi_constraints_upto5 1.0``). Each row becomes a verl sample:

    data_source  = "ifeval"                       # dispatch key (== open-instruct `dataset`)
    prompt       = [{"role": "user", "content": ...}]   # the instruction + constraints
    reward_model = {"style": "rule", "ground_truth": <stringified constraint list>}
    extra_info   = {"index", "key", "constraint", "constraint_type"}
    ability      = "instruction_following"

``reward_model.ground_truth`` is kept verbatim as the string-encoded
``[{'instruction_id': [...], 'kwargs': [...]}]`` that the IFEval verifier consumes.

Usage:
    python if_rlvr/convert_data.py \
        --output-dir /lustre/justinseo/if-verl/data/ifeval_multi --val-size 512 --seed 1
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset

HF_DATASET = "allenai/IF_multi_constraints_upto5"
DATA_SOURCE = "ifeval"


def to_verl_row(example: dict, idx: int) -> dict:
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    return {
        "data_source": DATA_SOURCE,
        "prompt": messages,
        "ability": "instruction_following",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["ground_truth"],  # string-encoded IFEval constraint spec
        },
        "extra_info": {
            "index": idx,
            "key": example.get("key", ""),
            "constraint": example.get("constraint", ""),
            "constraint_type": example.get("constraint_type", ""),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/lustre/justinseo/if-verl/data/ifeval_multi")
    ap.add_argument("--hf-dataset", default=HF_DATASET)
    ap.add_argument("--val-size", type=int, default=512, help="held-out validation examples (no train overlap)")
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed (open-instruct used --seed 1)")
    ap.add_argument("--max-train", type=int, default=-1, help="optional cap on train size (debug)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ds = load_dataset(args.hf_dataset, split="train")
    print(f"loaded {len(ds)} rows from {args.hf_dataset}")

    ds = ds.shuffle(seed=args.seed)
    val = ds.select(range(args.val_size))
    train = ds.select(range(args.val_size, len(ds)))
    if args.max_train > 0:
        train = train.select(range(min(args.max_train, len(train))))

    train_rows = [to_verl_row(train[i], i) for i in range(len(train))]
    val_rows = [to_verl_row(val[i], i) for i in range(len(val))]

    from datasets import Dataset

    train_out = os.path.join(args.output_dir, "train.parquet")
    val_out = os.path.join(args.output_dir, "val.parquet")
    Dataset.from_list(train_rows).to_parquet(train_out)
    Dataset.from_list(val_rows).to_parquet(val_out)
    print(f"wrote {len(train_rows)} train -> {train_out}")
    print(f"wrote {len(val_rows)} val   -> {val_out}")

    # sanity: show one row
    import json

    r = train_rows[0]
    print("\n=== sample verl row ===")
    print("data_source:", r["data_source"])
    print("prompt[0].role:", r["prompt"][0]["role"], "| content[:120]:", r["prompt"][0]["content"][:120])
    print("reward_model.ground_truth:", r["reward_model"]["ground_truth"][:200])
    print("extra_info:", json.dumps(r["extra_info"])[:160])


if __name__ == "__main__":
    main()
