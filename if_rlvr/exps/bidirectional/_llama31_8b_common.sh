#!/usr/bin/env bash
# Shared runtime defaults for meta-llama/Llama-3.1-8B-Instruct IF-RLVR jobs.
# This file is sourced by the precompute and GRPO launchers.

if [[ -n "${IF_RLVR_LLAMA31_COMMON_SH:-}" ]]; then
    return 0
fi
export IF_RLVR_LLAMA31_COMMON_SH=1

LLAMA31_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${LLAMA31_SCRIPT_DIR}/../../.." && pwd)}
export IFIF_ROOT=${IFIF_ROOT:-$(cd -- "${VERL_DIR}/.." && pwd)}
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${IFIF_ROOT}/IFBench/.nltk_data}

# Llama 3.1 is supported by the existing verl environment; unlike OLMo3 it
# does not need a separate Transformers overlay.
export LLAMA31_PYTHON_BIN=${LLAMA31_PYTHON_BIN:-${IFIF_ROOT}/.miniforge3/envs/verl/bin/python}
if [[ -x "${LLAMA31_PYTHON_BIN}" ]]; then
    export PATH="$(dirname -- "${LLAMA31_PYTHON_BIN}"):${PATH}"
fi
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Keep gated-model downloads in the shared experiment cache.
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
mkdir -p "${HF_HUB_CACHE}"

export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export MODEL_PATH=${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-${MODEL_PATH}}

# Llama-3.1-8B-Instruct has no reasoning/non-reasoning switch or </think>
# section. Do not pass Qwen's enable_thinking kwarg, and treat the complete
# generated response as the final answer for anchor PPL scoring.
export ENABLE_THINKING=false
export IF_APPLY_ENABLE_THINKING_KWARG=false
export IF_REQUIRE_THINK_END_FOR_REWARD=false
export IF_ALLOW_MISSING_THINK_FINAL_ANSWER=true

export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher_llama31_8b_instruct_nonreason_train_seed${IF_DATA_SEED}_val${IF_VAL_SIZE}_scored_by_llama31_8b_instruct.json}

# Match the established 8B/B200 IF-RLVR rollout profile. Sampling remains the
# verl GRPO default (temperature=1, top_p=1, no top-k filtering).
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-40960}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}

llama31_prepare_model_snapshot() {
    if [[ ! -x "${LLAMA31_PYTHON_BIN}" ]]; then
        echo "Missing verl Python runtime: ${LLAMA31_PYTHON_BIN}" >&2
        return 1
    fi

    # Resolve/download once in the launcher process before Ray starts. This
    # avoids four trainer/rollout workers racing on a gated Hub repository.
    local resolved_model_path
    resolved_model_path=$("${LLAMA31_PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_path = os.environ["MODEL_PATH"]
candidate = Path(model_path).expanduser()
if candidate.exists():
    print(candidate.resolve())
    raise SystemExit(0)

try:
    snapshot = Path(
        snapshot_download(
            repo_id=model_path,
            cache_dir=os.environ["HF_HUB_CACHE"],
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.tiktoken",
                "*.txt",
                "*.py",
            ],
            ignore_patterns=["original/*", "*.pth", "*.pt"],
        )
    )
except Exception as exc:
    raise SystemExit(
        f"Cannot download {model_path}: {type(exc).__name__}: {exc}\n"
        "This is a gated model. Accept the Meta Llama license and provide an "
        "authorized HF_TOKEN (or run `hf auth login`) before launching."
    ) from exc

required = [snapshot / "config.json", snapshot / "tokenizer_config.json"]
missing = [str(path) for path in required if not path.is_file()]
index_path = snapshot / "model.safetensors.index.json"
if index_path.is_file():
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    missing.extend(
        str(snapshot / shard)
        for shard in shards
        if not (snapshot / shard).is_file() or (snapshot / shard).stat().st_size == 0
    )
else:
    shards = list(snapshot.glob("*.safetensors"))
    if not shards or any(path.stat().st_size == 0 for path in shards):
        missing.append(str(snapshot / "*.safetensors"))
if missing:
    raise SystemExit("Incomplete Llama snapshot; missing: " + ", ".join(missing))

print(snapshot.resolve())
PY
    )
    export MODEL_PATH="${resolved_model_path}"
    echo "[Llama 3.1 snapshot] MODEL_PATH=${MODEL_PATH}" >&2
}

llama31_require_runtime() {
    if [[ ! -x "${LLAMA31_PYTHON_BIN}" ]]; then
        echo "Missing verl Python runtime: ${LLAMA31_PYTHON_BIN}" >&2
        return 1
    fi

    "${LLAMA31_PYTHON_BIN}" - <<'PY'
import os

import transformers
import vllm
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM  # noqa: F401
from vllm.model_executor.models.registry import ModelRegistry

model_path = os.environ["MODEL_PATH"]
if "LlamaForCausalLM" not in ModelRegistry.get_supported_archs():
    raise SystemExit(f"vLLM {vllm.__version__} does not register LlamaForCausalLM")

try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
except Exception as exc:
    raise SystemExit(
        f"Cannot access {model_path}: {type(exc).__name__}: {exc}\n"
        "meta-llama/Llama-3.1-8B-Instruct is gated. Make sure the account has "
        "accepted its license and set HF_TOKEN (or run `hf auth login`) before launching."
    ) from exc

architectures = set(getattr(config, "architectures", None) or [])
if getattr(config, "model_type", None) != "llama" or (
    architectures and "LlamaForCausalLM" not in architectures
):
    raise SystemExit(
        f"MODEL_PATH={model_path} is not a LlamaForCausalLM checkpoint: "
        f"model_type={getattr(config, 'model_type', None)!r}, architectures={sorted(architectures)!r}"
    )
if not getattr(tokenizer, "chat_template", None):
    raise SystemExit(f"Tokenizer for {model_path} has no chat_template")

rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": "runtime preflight"}],
    add_generation_prompt=True,
    tokenize=False,
)
if not rendered:
    raise SystemExit(f"Tokenizer for {model_path} produced an empty chat prompt")

print(
    f"[Llama 3.1 preflight] transformers={transformers.__version__} "
    f"vllm={vllm.__version__} architecture=LlamaForCausalLM model={model_path}"
)
PY
}
