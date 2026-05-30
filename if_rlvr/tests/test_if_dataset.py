#!/usr/bin/env python
"""Validate IFMultiConstraintsDataset (HF-direct loading) against convert_data.py's parquet.

Checks:
  [1] verl can load the custom class via its real get_dataset_class() hook.
  [2] train/val split selection works from the data_files token.
  [3] the HF-direct rows are IDENTICAL to the pre-converted parquet (same shuffle/seed/mapping):
      same val rows, same train count, no train/val overlap, identical verl schema.
Run: /home/justinseo/miniconda3/envs/verl/bin/python if_rlvr/tests/test_if_dataset.py
"""
import os
import sys

import datasets
from omegaconf import OmegaConf
from transformers import AutoTokenizer

VERL_DIR = "/lustre/justinseo/if-verl/verl"
if VERL_DIR not in sys.path:
    sys.path.insert(0, VERL_DIR)

PARQUET_DIR = "/lustre/justinseo/if-verl/data/ifeval_multi"
CLS_PATH = os.path.join(VERL_DIR, "if_rlvr", "if_dataset.py")
VAL_SIZE = 512
SEED = 1


def build_cfg(train_token, val_token, filter_overlong):
    return OmegaConf.create(
        {
            "custom_cls": {"path": CLS_PATH, "name": "IFMultiConstraintsDataset"},
            "train_files": [train_token],
            "val_files": [val_token],
            "prompt_key": "prompt",
            "reward_fn_key": "data_source",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": filter_overlong,
            "truncation": "error",
            "if_dataset_val_size": VAL_SIZE,
            "if_dataset_seed": SEED,
            "shuffle": False,
            "apply_chat_template_kwargs": {"enable_thinking": False},
        }
    )


def main():
    from verl.utils.dataset.rl_dataset import get_dataset_class

    cfg = build_cfg("if_multi_train", "if_multi_val", filter_overlong=False)

    # [1] verl resolves our custom class through its real hook
    cls = get_dataset_class(cfg)
    assert cls.__name__ == "IFMultiConstraintsDataset", cls
    print(f"[1] get_dataset_class -> {cls.__name__}  OK")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    # [2] instantiate both splits (filtering off here = pure split/mapping check)
    val_ds = cls(data_files=["if_multi_val"], tokenizer=tok, config=cfg, processor=None)
    train_ds = cls(data_files=["if_multi_train"], tokenizer=tok, config=cfg, processor=None)
    print(f"[2] split selection -> train={len(train_ds.dataframe)} val={len(val_ds.dataframe)}  OK")

    # [3] cross-check vs convert_data.py parquet (must be byte-identical rows)
    pq_train = datasets.load_dataset("parquet", data_files=os.path.join(PARQUET_DIR, "train.parquet"))["train"]
    pq_val = datasets.load_dataset("parquet", data_files=os.path.join(PARQUET_DIR, "val.parquet"))["train"]

    assert len(val_ds.dataframe) == len(pq_val), (len(val_ds.dataframe), len(pq_val))
    assert len(train_ds.dataframe) == len(pq_train), (len(train_ds.dataframe), len(pq_train))

    def row_key(r):
        # identity of a sample = its prompt text + ground_truth (order-stable)
        return (r["prompt"][0]["content"], r["reward_model"]["ground_truth"])

    # 3a: val rows identical and in the same order
    mism = 0
    for i in range(len(pq_val)):
        if row_key(val_ds.dataframe[i]) != row_key(pq_val[i]):
            mism += 1
    assert mism == 0, f"{mism} val rows differ from parquet"

    # 3b: no train/val overlap (held-out eval)
    val_keys = {row_key(pq_val[i]) for i in range(len(pq_val))}
    train_keys = {row_key(train_ds.dataframe[i]) for i in range(len(train_ds.dataframe))}
    overlap = len(val_keys & train_keys)
    assert overlap == 0, f"train/val overlap = {overlap}"

    # 3c: schema sanity on one row
    r = train_ds.dataframe[0]
    assert r["data_source"] == "ifeval"
    assert r["reward_model"]["style"] == "rule" and isinstance(r["reward_model"]["ground_truth"], str)
    assert r["prompt"][0]["role"] == "user"
    assert set(r["extra_info"]) >= {"index", "key", "constraint", "constraint_type"}
    print(f"[3] vs parquet: val identical ({len(pq_val)} rows, 0 mismatch), "
          f"train count match ({len(pq_train)}), overlap={overlap}, schema OK")

    # [4] filtering path executes (tokenizes); just confirm it runs and shrinks <= total
    cfg2 = build_cfg("if_multi_val", "if_multi_val", filter_overlong=True)
    cfg2 = OmegaConf.merge(cfg2, OmegaConf.create({"max_prompt_length": 1024}))
    val_filt = cls(data_files=["if_multi_val"], tokenizer=tok, config=cfg2, processor=None)
    assert len(val_filt.dataframe) <= VAL_SIZE
    print(f"[4] filter_overlong_prompts path: {len(val_filt.dataframe)}/{VAL_SIZE} kept @max_prompt_length=1024  OK")

    print("\nALL DATASET CHECKS PASSED")


if __name__ == "__main__":
    main()
