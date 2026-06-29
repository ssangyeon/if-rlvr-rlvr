#!/usr/bin/env python3
"""Fail early when the active environment cannot run Gemma 4 models."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from typing import Any


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _load_config_json(model_name: str) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:  # noqa: BLE001
        print(f"[Gemma4 preflight] Could not read config.json for {model_name}: {exc}", file=sys.stderr)
        return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH", "")
    if not model_name:
        print("[Gemma4 preflight] MODEL_PATH is empty.", file=sys.stderr)
        return 2

    infer_backend = os.environ.get("INFER_BACKEND", "vllm").strip().lower() or "vllm"
    failures: list[str] = []
    config_json = _load_config_json(model_name)
    expected_archs = _as_list((config_json or {}).get("architectures"))
    expected_model_type = (config_json or {}).get("model_type")
    expected_tf_version = (config_json or {}).get("transformers_version")

    print(
        "[Gemma4 preflight] model="
        f"{model_name} expected_model_type={expected_model_type} "
        f"expected_architectures={expected_archs or 'unknown'} "
        f"transformers={_version('transformers')} vllm={_version('vllm')}"
    )

    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        failures.append(f"transformers import failed: {exc}")
    else:
        try:
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
            expected_archs = _as_list(getattr(hf_config, "architectures", None)) or expected_archs
            expected_model_type = getattr(hf_config, "model_type", expected_model_type)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                "AutoConfig cannot load this Gemma4 model: "
                f"{type(exc).__name__}: {exc}. "
                f"Model metadata expects transformers_version={expected_tf_version}."
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            if not getattr(tokenizer, "chat_template", None):
                failures.append("AutoTokenizer loaded, but chat_template is missing.")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"AutoTokenizer cannot load this Gemma4 model: {type(exc).__name__}: {exc}")

    if infer_backend == "vllm":
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            supported_archs = set(ModelRegistry.get_supported_archs())
            if expected_archs and not any(arch in supported_archs for arch in expected_archs):
                failures.append(
                    "vLLM does not advertise support for "
                    f"{expected_archs}; install a vLLM build that includes Gemma4 support."
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"vLLM registry check failed: {type(exc).__name__}: {exc}")

    if failures:
        print("[Gemma4 preflight] FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "[Gemma4 preflight] Fix the active conda/env first, or set "
            "IF_GEMMA4_PREFLIGHT=false only if you have already patched runtime support.",
            file=sys.stderr,
        )
        return 2

    print("[Gemma4 preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
