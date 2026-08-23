#!/usr/bin/env python3
"""Materialize one ``global_step_<N>`` folder of a verl-exported HF checkpoint
repo (as written by ``push_checkpoints_to_hf.py``) to a local directory, and
print the resulting local path on stdout.

Used to warm-start a NEW trainer instance (fresh optimizer state, e.g. no
local FSDP checkpoint survives to do a real ``trainer.resume_mode=resume_path``
resume) from an already-trained checkpoint that only exists on the Hub as a
bf16 HF export: point ``MODEL_PATH`` at the printed directory instead of a
bare repo id.

    path=$(download_hf_checkpoint_step.py --repo-id org/run --step 364 --local-dir /cache/dir)
    export MODEL_PATH="${path}"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def log(message: str) -> None:
    print(f"[download-checkpoint] {message}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Hub model repo holding global_step_<N>/ folders")
    parser.add_argument("--step", required=True, type=int, help="which global_step_<N> to fetch")
    parser.add_argument("--local-dir", required=True, help="local cache root; the step lands at <local-dir>/global_step_<N>")
    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    step_name = f"global_step_{args.step}"
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    target = local_dir / step_name
    if target.is_dir() and (target / "config.json").is_file() and (target / "model.safetensors.index.json").is_file():
        log(f"already present: {target}")
        print(str(target.resolve()))
        return 0

    log(f"downloading {args.repo_id}:{step_name} -> {local_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=[f"{step_name}/*"],
    )

    if not (target / "config.json").is_file():
        log(f"ERROR: {target}/config.json missing after download; {args.repo_id} may not have {step_name}")
        return 1
    index = target / "model.safetensors.index.json"
    if index.is_file():
        import json

        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        missing = [name for name in shards if not (target / name).is_file() or (target / name).stat().st_size == 0]
        if missing:
            log(f"ERROR: {target} missing shards: {missing}")
            return 1
    log(f"ready: {target}")
    print(str(target.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
