#!/usr/bin/env python3
"""Push every completed verl actor checkpoint to a Hugging Face Hub model repo.

When ``actor_rollout_ref.actor.checkpoint.save_contents`` includes ``hf_model``,
verl writes a self-contained Hugging Face model to

    <ckpt_dir>/global_step_<N>/actor/huggingface/

and only writes ``<ckpt_dir>/latest_checkpointed_iteration.txt`` once the whole
step has been persisted (see ``RayPPOTrainer._save_checkpoint``).  This watcher
polls that marker, so it never uploads a half-written checkpoint.

Each step lands in its own repo folder, so one repo holds the whole run:

    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(repo_id, subfolder="global_step_91")

Typical use (daemon alongside training, plus a final sweep on exit):

    push_checkpoints_to_hf.py --ckpt-dir CKPT --repo-id user/run --private &
    push_checkpoints_to_hf.py --ckpt-dir CKPT --repo-id user/run --once
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi

LATEST_MARKER = "latest_checkpointed_iteration.txt"
STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")

README_TEMPLATE = """---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- verl
- grpo
- rlvr
- instruction-following
---

# {run_name}

GRPO checkpoints exported by [verl](https://github.com/volcengine/verl).

Every `global_step_<N>/` folder is a complete bfloat16 Hugging Face model,
uploaded as soon as the trainer finished writing that step.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

step = "global_step_{first_step}"
model = AutoModelForCausalLM.from_pretrained("{repo_id}", subfolder=step)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", subfolder=step)
```

- Base model: `{base_model}`
- Run name: `{run_name}`
"""


def log(message: str) -> None:
    print(f"[hf-push] {message}", flush=True)


def read_latest_step(ckpt_dir: Path) -> int:
    """Return the newest fully written step, or -1 when the marker is absent."""
    marker = ckpt_dir / LATEST_MARKER
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return -1


def discover_steps(ckpt_dir: Path) -> list[int]:
    if not ckpt_dir.is_dir():
        return []
    steps = []
    for entry in ckpt_dir.iterdir():
        match = STEP_DIR_RE.match(entry.name)
        if match and entry.is_dir():
            steps.append(int(match.group(1)))
    return sorted(steps)


def hf_model_dir(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"global_step_{step}" / "actor" / "huggingface"


def missing_weight_files(folder: Path) -> list[str]:
    """Return the weight files the folder promises but does not have."""
    if not (folder / "config.json").is_file():
        return ["config.json"]

    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, ValueError) as exc:
            return [f"model.safetensors.index.json ({exc})"]
        shards = sorted(set(weight_map.values()))
        if not shards:
            return ["model.safetensors.index.json (empty weight_map)"]
        return [name for name in shards if not (folder / name).is_file() or (folder / name).stat().st_size == 0]

    single = folder / "model.safetensors"
    if single.is_file() and single.stat().st_size > 0:
        return []
    return ["model.safetensors or model.safetensors.index.json"]


def load_state(state_file: Path) -> set[int]:
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {int(step) for step in payload.get("uploaded", [])}


def save_state(state_file: Path, uploaded: set[int]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"uploaded": sorted(uploaded)}
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_hf_push_", suffix=".json", dir=str(state_file.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, state_file)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def ensure_repo(api: HfApi, repo_id: str, private: bool) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def ensure_readme(api: HfApi, repo_id: str, base_model: str, run_name: str, first_step: int) -> None:
    """Write a minimal model card once, so the repo is self-describing."""
    try:
        if api.file_exists(repo_id=repo_id, filename="README.md", repo_type="model"):
            return
    except Exception:  # noqa: BLE001 - never let the card block a weight upload
        return
    card = README_TEMPLATE.format(
        base_model=base_model or "unknown",
        run_name=run_name or repo_id.split("/")[-1],
        repo_id=repo_id,
        first_step=first_step,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add run card",
    )


def upload_step(api: HfApi, repo_id: str, folder: Path, step: int, run_name: str) -> None:
    path_in_repo = f"global_step_{step}"
    log(f"uploading {folder} -> {repo_id}:{path_in_repo}")
    started = time.monotonic()
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"{run_name or 'verl'}: global_step_{step}",
    )
    log(f"uploaded global_step_{step} in {time.monotonic() - started:.0f}s")


def sweep(args: argparse.Namespace, api: HfApi, uploaded: set[int]) -> bool:
    """Upload every complete-but-unsent step. Returns True when state changed."""
    ckpt_dir = Path(args.ckpt_dir)
    latest = read_latest_step(ckpt_dir)
    if latest < 0:
        return False

    changed = False
    for step in discover_steps(ckpt_dir):
        if step in uploaded or step > latest:
            continue
        folder = hf_model_dir(ckpt_dir, step)
        if not folder.is_dir():
            log(f"skipping global_step_{step}: no hf_model export at {folder}")
            continue
        missing = missing_weight_files(folder)
        if missing:
            log(f"global_step_{step} not ready yet, missing: {', '.join(missing[:4])}")
            continue

        for attempt in range(1, args.max_retries + 1):
            try:
                ensure_repo(api, args.repo_id, args.private)
                ensure_readme(api, args.repo_id, args.base_model, args.run_name, step)
                upload_step(api, args.repo_id, folder, step, args.run_name)
                uploaded.add(step)
                save_state(Path(args.state_file), uploaded)
                changed = True
                break
            except Exception:  # noqa: BLE001 - the trainer must never die because of an upload
                log(f"upload of global_step_{step} failed (attempt {attempt}/{args.max_retries}):")
                traceback.print_exc()
                if attempt < args.max_retries:
                    time.sleep(args.retry_seconds)
    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True, help="verl trainer.default_local_dir for this run")
    parser.add_argument("--repo-id", required=True, help="target Hub model repo, e.g. user/my-run")
    parser.add_argument("--state-file", default=None, help="upload bookkeeping (default: <ckpt-dir>/.hf_push_state.json)")
    parser.add_argument("--run-name", default="", help="experiment name, used in commit messages and the card")
    parser.add_argument("--base-model", default="", help="base model id recorded in the generated model card")
    parser.add_argument("--private", action="store_true", help="create the repo private when it does not exist")
    parser.add_argument("--once", action="store_true", help="sweep once and exit instead of watching")
    parser.add_argument("--poll-seconds", type=float, default=120.0, help="watch-mode poll interval")
    parser.add_argument("--max-retries", type=int, default=5, help="upload attempts per checkpoint")
    parser.add_argument("--retry-seconds", type=float, default=60.0, help="delay between upload attempts")
    args = parser.parse_args(argv)
    if args.state_file is None:
        args.state_file = str(Path(args.ckpt_dir) / ".hf_push_state.json")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api = HfApi(token=os.getenv("HF_TOKEN") or None)
    uploaded = load_state(Path(args.state_file))
    log(f"watching {args.ckpt_dir} -> https://huggingface.co/{args.repo_id} (already uploaded: {sorted(uploaded)})")

    if args.once:
        sweep(args, api, uploaded)
        return 0

    while True:
        try:
            sweep(args, api, uploaded)
        except Exception:  # noqa: BLE001 - the watcher outlives any single failure
            traceback.print_exc()
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
