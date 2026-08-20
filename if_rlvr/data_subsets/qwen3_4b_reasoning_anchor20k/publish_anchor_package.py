#!/usr/bin/env python3
"""Publish the complete subset20k anchor package without persisting a token."""

from __future__ import annotations

import argparse
import collections
import getpass
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


DEFAULT_REPO = "sangyon/anchor_cache"
DEFAULT_PREFIX = "qwen3_4b_reasoning_anchor20k_n3"
CODE_BASE_REVISION = "af34fac4b2dae646d0ddf5afa6dcfad7b4cc0745"
EXPECTED_CATEGORY_COUNTS = {
    "docs": 3,
    "artifacts": 10,
    "manifests": 25,
    "code": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--token-stdin",
        action="store_true",
        help="read one token line from stdin instead of HF_TOKEN or a hidden prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, hash, and list the exact public payload without network access",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_token(token_stdin: bool) -> str:
    if token_stdin:
        token = sys.stdin.readline().strip()
    else:
        token = os.environ.get("HF_TOKEN", "").strip() or getpass.getpass("HF token: ").strip()
    if not token:
        raise RuntimeError("no Hugging Face token supplied")
    return token


def collect_files(workspace: Path, prefix: str) -> dict[str, Path]:
    source = workspace / "if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k"
    runtime = workspace / ".agent_runtime/subset20k"
    files: dict[str, Path] = {}

    def add(remote: str, local: Path) -> None:
        if not local.is_file():
            raise FileNotFoundError(local)
        if remote in files:
            raise RuntimeError(f"duplicate remote path: {remote}")
        files[remote] = local

    add(f"{prefix}/README.md", source / "HUGGINGFACE_GENERATION_GUIDE.md")
    add(f"{prefix}/CURATION_AND_REPRODUCTION.md", source / "README.md")
    add(f"{prefix}/current_generation_status.json", source / "current_generation_status.json")

    for name in ("selected_train.parquet", "selected_audit.parquet", "audit_frame.parquet"):
        add(f"{prefix}/artifacts/{name}", runtime / name)
    for name in (
        "run1.SUBSET20480.AVAILABLE.json",
        "run1.SUBSET20480.best_available.json",
        "run2.SUBSET20480.AVAILABLE.json",
        "run3.SUBSET20480.AVAILABLE.json",
    ):
        add(f"{prefix}/artifacts/anchor_views/{name}", runtime / "anchor_views" / name)
    add(
        f"{prefix}/artifacts/generated_runs/run1.missing.cache.json",
        runtime / "generated_runs/run1.missing.cache.json",
    )
    for name in ("generation_plan.json", "audit_provenance.json"):
        add(f"{prefix}/artifacts/{name}", runtime / name)

    for name in (
        "anchor_coverage.json",
        "provenance.json",
        "verification_report.md",
        "subset_indices.json",
        "train_indices.json",
        "panel_A_indices.json",
        "panel_B_indices.json",
        "panel_C_indices.json",
        "panel_D_indices.json",
        "panel_E_indices.json",
    ):
        add(f"{prefix}/manifests/{name}", source / name)
    for local in sorted((source / "generation_manifests").glob("run*/*.json")):
        add(
            f"{prefix}/manifests/generation_manifests/{local.parent.name}/{local.name}",
            local,
        )

    for name in (
        "build_audit_frame.py",
        "curate_anchor_subset.py",
        "prepare_anchor_views.py",
        "run_anchor_generation_20k.sh",
        "generate_missing_run.sh",
        "generate_missing_shard.sh",
        "publish_anchor_package.py",
    ):
        add(f"{prefix}/code/{name}", source / name)
    return files


def validate_publication_scope(files: dict[str, Path], workspace: Path, prefix: str) -> None:
    category_counts: collections.Counter[str] = collections.Counter()
    for remote, local in files.items():
        relative = remote.removeprefix(prefix + "/")
        if remote == f"{prefix}/README.md" or remote == f"{prefix}/CURATION_AND_REPRODUCTION.md":
            category = "docs"
        elif remote == f"{prefix}/current_generation_status.json":
            category = "docs"
        else:
            category = relative.split("/", 1)[0]
        category_counts[category] += 1
        if not remote.startswith(prefix + "/"):
            raise RuntimeError(f"path escapes artifact prefix: {remote}")
        if local.is_symlink():
            raise RuntimeError(f"refusing symlink: {local}")
        try:
            local.resolve().relative_to(workspace)
        except ValueError as exc:
            raise RuntimeError(f"local source escapes workspace: {local}") from exc
        lowered = local.name.lower()
        if any(marker in lowered for marker in ("token", "credential", "secret", "wandb")):
            raise RuntimeError(f"sensitive-looking filename is out of scope: {local}")
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise RuntimeError(
            f"publication scope changed: {dict(category_counts)} != {EXPECTED_CATEGORY_COUNTS}"
        )


def validate_no_embedded_credentials(files: dict[str, Path]) -> None:
    patterns = {
        "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
        "wandb_key_assignment": re.compile(
            rb"WANDB_API_KEY\s*=\s*[\"']?[A-Za-z0-9]{30,}"
        ),
    }
    for local in files.values():
        overlap = b""
        with local.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                window = overlap + block
                for label, pattern in patterns.items():
                    if pattern.search(window):
                        raise RuntimeError(f"{label} pattern detected in public payload: {local}")
                overlap = window[-256:]


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    files = collect_files(workspace, args.prefix)
    validate_publication_scope(files, workspace, args.prefix)
    validate_no_embedded_credentials(files)
    records = [
        {
            "path": remote.removeprefix(args.prefix + "/"),
            "bytes": local.stat().st_size,
            "sha256": sha256_file(local),
        }
        for remote, local in sorted(files.items())
    ]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_VALID",
                    "network_used": False,
                    "artifact_prefix": args.prefix,
                    "category_counts": EXPECTED_CATEGORY_COUNTS,
                    "file_count": len(records),
                    "total_bytes": sum(record["bytes"] for record in records),
                    "files": records,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    api = HfApi(token=read_token(args.token_stdin))
    identity = api.whoami()
    repo = api.repo_info(args.repo_id, repo_type="dataset")
    manifest = {
        "version": 1,
        "repo_id": args.repo_id,
        "artifact_prefix": args.prefix,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upload_parent_commit": repo.sha,
        "uploader": identity.get("name") or identity.get("fullname") or "authenticated-user",
        "code_base_git_revision": CODE_BASE_REVISION,
        "file_count_excluding_this_manifest": len(records),
        "total_bytes_excluding_this_manifest": sum(record["bytes"] for record in records),
        "files": records,
    }
    manifest_file = io.BytesIO((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
        for remote, local in sorted(files.items())
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"{args.prefix}/FILE_MANIFEST.json",
            path_or_fileobj=manifest_file,
        )
    )
    print(
        f"authenticated_as={manifest['uploader']} parent={repo.sha} "
        f"files={len(operations)} bytes={manifest['total_bytes_excluding_this_manifest']}",
        flush=True,
    )
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish Qwen3-4B subset20k anchor data and generation guide",
        commit_description=(
            "Includes the complete selected data/audit artifacts, current run1 data, "
            "exact run2/run3 missing manifests, checksums, and validated generation source."
        ),
        parent_commit=repo.sha,
    )
    print(f"commit_url={result.commit_url}", flush=True)
    print(f"commit_oid={result.oid}", flush=True)


if __name__ == "__main__":
    main()
