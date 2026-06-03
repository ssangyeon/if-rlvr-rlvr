#!/usr/bin/env python
"""verl custom dataset: load ``allenai/IF_multi_constraints_upto5`` straight from the HF Hub.

This mirrors open-instruct's ``scripts/train/rlvr/valpy_if_grpo_fast.sh``, which pulls the
dataset directly from the Hub (``--dataset_mixer_list allenai/IF_multi_constraints_upto5 1.0``)
with no local parquet-conversion step. Each HF row is mapped on the fly into verl's RLVR schema,
producing *exactly* the rows ``if_rlvr/convert_data.py`` would have written to parquet — so the
reward path, ground-truth encoding, and train/val split are byte-for-byte identical; only the
source changes (Hub instead of local parquet).

Wire it in (already done in the run scripts):
    data.custom_cls.path=<this file>
    data.custom_cls.name=IFMultiConstraintsDataset
    data.train_files="['if_multi_train']"     # the token selects the split (see below)
    data.val_files="['if_multi_val']"

Split selection: verl instantiates this same class once with ``data.train_files`` and once with
``data.val_files`` (see verl/trainer/main_ppo.py:create_rl_dataset). We inspect the passed token:
if it contains ``val``/``eval``/``valid``/``test`` it is the held-out eval split, else train.

Tunables (read from ``config.data.*``, all optional):
    if_dataset_hf        (default allenai/IF_multi_constraints_upto5)
    if_dataset_val_size  (default 512)   held-out eval examples, NO train overlap
    if_dataset_seed      (default 1)     shuffle seed (open-instruct used --seed 1)

Note: open-instruct's eval list samples 16 examples *from the train split* (overlapping). We
instead hold out ``if_dataset_val_size`` rows with no train overlap (a cleaner eval); set
``+data.if_dataset_val_size=16`` if you want to match open-instruct's eval size.

This class subclasses verl's RLHFDataset (the verl-sanctioned ``data.custom_cls`` extension
point), so it necessarily imports verl — unlike the reward core, which stays verl-independent.
It is still additive: it modifies no existing verl file.
"""

from __future__ import annotations

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset

HF_DATASET = "allenai/IF_multi_constraints_upto5"
DATA_SOURCE = "ifeval"
_VAL_MARKERS = ("val", "eval", "valid", "test")


def _make_constraint_free_messages(example: dict) -> list[dict]:
    # ##6/3 ppl## Build x for p(y|x): remove the dataset constraint text from the user prompt.
    constraint = (example.get("constraint") or "").strip()
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    if not constraint:
        return messages
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if constraint in content:
            message["content"] = content.replace(constraint, "", 1).strip()
            break
    return messages


def _to_verl_row(example: dict, idx: int) -> dict:
    """Map one HF row to verl's RLVR schema (identical to convert_data.py:to_verl_row)."""
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    return {
        "data_source": DATA_SOURCE,
        "prompt": messages,
        # ##6/3 ppl## Carried to the agent loop for vLLM scoring of final-answer p(y|x).
        "ppl_prompt": _make_constraint_free_messages(example),
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


class IFMultiConstraintsDataset(RLHFDataset):
    """RLHFDataset that sources its rows from the HF Hub instead of local parquet files."""

    def _download(self, use_origin_parquet=False):
        # Data comes from the HF Hub (handled in _read_files_and_tokenize); nothing to copy.
        return

    def _read_files_and_tokenize(self):
        cfg = self.config
        hf_name = cfg.get("if_dataset_hf", HF_DATASET)
        val_size = int(cfg.get("if_dataset_val_size", 512))
        seed = int(cfg.get("if_dataset_seed", 1))

        # Which split does THIS instance want? verl passes data.train_files for train and
        # data.val_files for val; we read the token to decide (see module docstring).
        tokens = " ".join(str(f) for f in self.original_data_files).lower()
        want_val = any(m in tokens for m in _VAL_MARKERS)

        ds = datasets.load_dataset(hf_name, split="train")
        ds = ds.shuffle(seed=seed)  # deterministic; identical split to convert_data.py
        if want_val:
            ds = ds.select(range(val_size))
        else:
            ds = ds.select(range(val_size, len(ds)))

        rows = [_to_verl_row(ds[i], i) for i in range(len(ds))]
        self.dataframe: datasets.Dataset = datasets.Dataset.from_list(rows)

        total = len(self.dataframe)
        print(
            f"[IFMultiConstraintsDataset] {'val' if want_val else 'train'} split: "
            f"{total} rows from {hf_name} (seed={seed}, val_size={val_size})"
        )

        # Preserve RLHFDataset's max_samples sub-sampling + long-prompt filtering behavior.
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(*((self.seed,) if self.seed is not None else ()))
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"[IFMultiConstraintsDataset] selected {self.max_samples} of {total} samples")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
