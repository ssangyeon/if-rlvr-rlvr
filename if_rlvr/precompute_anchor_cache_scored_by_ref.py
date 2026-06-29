#!/usr/bin/env python3
"""Rescore an IF ref-anchor cache with a different reference policy.

This is for experiments such as:

    0.6B reasoning policy + 4B non-reasoning anchor answers

The input cache supplies the anchor continuations (``y0``/``y1``). This script
keeps those continuations, but recomputes ``ref0_*`` and ``ref1_*`` likelihoods
with the scorer model, so anchor and policy PPL live in the same model space.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_DATASET = "allenai/IF_multi_constraints_upto5"


def _split_constraints(raw) -> list[str]:
    return [part.strip() for part in (raw or "").split("\t") if part.strip()]


def _strip_constraints_from_text(content: str, parts: list[str]) -> str | None:
    starts = [content.find(part) for part in parts]
    found = [pos for pos in starts if pos >= 0]
    if not found:
        return None
    base = content[: min(found)].strip()
    if base and not any(part in base for part in parts):
        return base
    base = content
    for part in sorted(parts, key=len, reverse=True):
        base = base.replace(part, " ")
    return re.sub(r"[ \t]{2,}", " ", base).strip()


def _make_constraint_free_messages(example: dict) -> list[dict]:
    parts = _split_constraints(example.get("constraint"))
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    if not parts:
        return messages
    for message in messages:
        if message.get("role") != "user":
            continue
        stripped = _strip_constraints_from_text(message.get("content", ""), parts)
        if stripped is not None:
            message["content"] = stripped
    return messages


def _load_train_prefix_ids(
    tokenizer,
    dataset_name: str,
    seed: int,
    val_size: int,
    enable_thinking: bool,
) -> dict[int, list[int]]:
    ds = datasets.load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(val_size, len(ds)))

    prefixes: dict[int, list[int]] = {}
    for idx in range(len(ds)):
        messages = _make_constraint_free_messages(ds[idx])
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(tokenized, dict):
            tokenized = tokenized.get("input_ids", [])
        prefixes[idx] = list(tokenized)
    return prefixes


def _dtype_from_name(name: str):
    table = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _score_batch(
    model,
    device: torch.device,
    pad_token_id: int,
    prefixes: list[list[int]],
    continuations: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not prefixes:
        empty_f = np.asarray([], dtype=np.float64)
        empty_i = np.asarray([], dtype=np.int64)
        return empty_f, empty_i, empty_f

    rows = []
    labels = []
    lengths = []
    max_len = 0
    for prefix, continuation in zip(prefixes, continuations, strict=True):
        prefix = list(prefix or [])
        continuation = list(continuation or [])
        ids = prefix + continuation
        if len(ids) < 2 or not continuation:
            ids = ids or [pad_token_id]
        label = list(ids)
        prefix_len = min(len(prefix), len(label))
        for i in range(prefix_len):
            label[i] = -100
        rows.append(ids)
        labels.append(label)
        lengths.append(len(continuation))
        max_len = max(max_len, len(ids))

    input_ids = []
    label_ids = []
    attention_mask = []
    for ids, label in zip(rows, labels, strict=True):
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        label_ids.append(label + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
    attention_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = label_tensor[:, 1:].contiguous()
        flat_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        token_loss = flat_loss.view(shift_labels.shape)
        valid = shift_labels.ne(-100)
        nll = (token_loss * valid).sum(dim=1).detach().cpu().to(torch.float64).numpy()
        counts = valid.sum(dim=1).detach().cpu().to(torch.int64).numpy()

    ppls = np.full_like(nll, np.inf, dtype=np.float64)
    ok = counts > 0
    ppls[ok] = np.exp(nll[ok] / counts[ok])
    return nll, counts, ppls


def _worker_main(
    rank: int,
    args_dict: dict[str, Any],
    entries: list[tuple[int, dict]],
    prefix_ids: dict[int, list[int]],
    partial_path: str,
) -> None:
    args = argparse.Namespace(**args_dict)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.scorer_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")

    model = AutoModelForCausalLM.from_pretrained(
        args.scorer_model,
        torch_dtype=_dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    result: dict[str, dict] = {}
    total = len(entries)
    for start in range(0, total, args.micro_batch_size):
        batch = entries[start : start + args.micro_batch_size]
        indices = [index for index, _ in batch]
        prefixes = [prefix_ids.get(index, []) for index in indices]
        y0 = [list(item.get("y0") or []) for _, item in batch]
        y1 = [list(item.get("y1") or []) for _, item in batch]

        ref0_nll, ref0_counts, ref0_ppls = _score_batch(model, device, pad_token_id, prefixes, y0)
        ref1_nll, ref1_counts, ref1_ppls = _score_batch(model, device, pad_token_id, prefixes, y1)

        for local_idx, (index, item) in enumerate(batch):
            result[str(index)] = {
                "y0": y0[local_idx],
                "y1": y1[local_idx],
                "ref0_nll": float(ref0_nll[local_idx]),
                "ref0_token_count": int(ref0_counts[local_idx]),
                "ref0_ppl": float(ref0_ppls[local_idx]),
                "ref1_nll": float(ref1_nll[local_idx]),
                "ref1_token_count": int(ref1_counts[local_idx]),
                "ref1_ppl": float(ref1_ppls[local_idx]),
            }
        if rank == 0 and (start == 0 or (start // args.micro_batch_size) % 20 == 0):
            print(f"[worker {rank}] scored {min(start + args.micro_batch_size, total)}/{total}")

    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def _metadata(args, tokenizer_class: str) -> dict[str, Any]:
    return {
        "version": 2,
        "model_path": args.scorer_model,
        "tokenizer_class": tokenizer_class,
        "train_files": ["if_multi_train"],
        "if_dataset_hf": "" if args.dataset_name == HF_DATASET else args.dataset_name,
        "if_dataset_seed": args.seed,
        "if_dataset_val_size": args.val_size,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "apply_chat_template_kwargs": {"enable_thinking": args.metadata_enable_thinking},
        "rollout_temperature": args.rollout_temperature,
        "rollout_top_p": args.rollout_top_p,
        "rollout_top_k": args.rollout_top_k,
        "response_length": args.response_length,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--scorer-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--anchor-answer-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset-name", default=HF_DATASET)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ppl-enable-thinking", action="store_true")
    parser.add_argument("--metadata-enable-thinking", action="store_true", default=False)
    parser.add_argument("--max-prompt-length", type=int, default=16384)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--response-length", type=int, default=8192)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_cache)
    output_path = Path(args.output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this precompute script")
    visible_gpus = torch.cuda.device_count()
    if args.num_workers > visible_gpus:
        raise RuntimeError(f"Requested {args.num_workers} workers but only {visible_gpus} CUDA devices are visible")

    print(f"[load] input cache: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        input_payload = json.load(f)
    input_items = input_payload.get("items", {})
    entries = sorted((int(key), item) for key, item in input_items.items())
    entries = [(index, item) for index, item in entries if item.get("y0") and item.get("y1")]
    print(f"[load] anchor answer entries: {len(entries)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.scorer_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print(f"[dataset] tokenizing p(y|x) prefixes with enable_thinking={args.ppl_enable_thinking}")
    prefix_ids = _load_train_prefix_ids(
        tokenizer,
        args.dataset_name,
        args.seed,
        args.val_size,
        args.ppl_enable_thinking,
    )

    chunks = [entries[i :: args.num_workers] for i in range(args.num_workers)]
    ctx = get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="if_anchor_rescore_") as tmpdir:
        partial_paths = [str(Path(tmpdir) / f"part_{rank}.json") for rank in range(args.num_workers)]
        procs = []
        args_dict = vars(args)
        for rank, chunk in enumerate(chunks):
            proc = ctx.Process(
                target=_worker_main,
                args=(rank, args_dict, chunk, prefix_ids, partial_paths[rank]),
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker pid={proc.pid} exited with code {proc.exitcode}")

        merged_items: dict[str, dict] = {}
        for path in partial_paths:
            with open(path, "r", encoding="utf-8") as f:
                merged_items.update(json.load(f))

    payload = {
        "metadata": _metadata(args, tokenizer.__class__.__name__),
        "source_metadata": {
            "input_cache": str(input_path),
            "input_metadata": input_payload.get("metadata", {}),
            "anchor_answer_model_path": args.anchor_answer_model,
            "scorer_model_path": args.scorer_model,
            "ppl_apply_chat_template_kwargs": {"enable_thinking": args.ppl_enable_thinking},
            "note": "y0/y1 copied from input cache; ref0/ref1 likelihoods rescored by scorer_model_path.",
        },
        "items": {key: merged_items[key] for key in sorted(merged_items, key=lambda value: int(value))},
    }

    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_if_anchor_rescore_", suffix=".json", dir=str(output_path.parent))
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_name, output_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    print(f"[done] wrote {len(merged_items)} rescored anchor items to {output_path}")


if __name__ == "__main__":
    main()
