#!/usr/bin/env bash
# Shared runtime defaults for allenai/Olmo-3-7B-Instruct-DPO IF-RLVR jobs.
# This file is sourced by the precompute and GRPO launchers.

if [[ -n "${IF_RLVR_OLMO3_COMMON_SH:-}" ]]; then
    return 0
fi
export IF_RLVR_OLMO3_COMMON_SH=1

OLMO3_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${OLMO3_SCRIPT_DIR}/../../.." && pwd)}
export IFIF_ROOT=${IFIF_ROOT:-$(cd -- "${VERL_DIR}/.." && pwd)}
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${IFIF_ROOT}/IFBench/.nltk_data}

# Keep OLMo3's Transformers 4.57.1 overlay separate from the Qwen/SGLang
# environment, whose optional runtime stack currently uses Transformers 4.56.1.
export OLMO3_VENV_DIR=${OLMO3_VENV_DIR:-${IFIF_ROOT}/.venvs/verl-olmo3}
export OLMO3_PYTHON_BIN=${OLMO3_PYTHON_BIN:-${OLMO3_VENV_DIR}/bin/python}
export OLMO3_BASE_PYTHON_BIN=${OLMO3_BASE_PYTHON_BIN:-${IFIF_ROOT}/.miniforge3/envs/verl/bin/python}

if [[ -x "${OLMO3_PYTHON_BIN}" ]]; then
    export PATH="$(dirname -- "${OLMO3_PYTHON_BIN}"):${PATH}"
fi

export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export MODEL_PATH=${MODEL_PATH:-allenai/Olmo-3-7B-Instruct-DPO}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-${MODEL_PATH}}

# Olmo-3-7B-Instruct-DPO is a non-thinking model. Its chat template does not
# implement Qwen's enable_thinking kwarg and does not emit </think> markers.
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-false}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}
export IF_ALLOW_MISSING_THINK_FINAL_ANSWER=${IF_ALLOW_MISSING_THINK_FINAL_ANSWER:-true}

export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher_olmo3_7b_instruct_dpo_nonreason_train_seed${IF_DATA_SEED}_val${IF_VAL_SIZE}_scored_by_olmo3_7b_instruct_dpo.json}

# Match the latest Qwen3-8B rollout profile used in this workspace.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-40960}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.6}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}

olmo3_require_runtime() {
    if [[ ! -x "${OLMO3_PYTHON_BIN}" ]]; then
        echo "Missing OLMo3 runtime: ${OLMO3_PYTHON_BIN}" >&2
        echo "Run ${OLMO3_SCRIPT_DIR}/setup_olmo3_runtime.sh first." >&2
        return 1
    fi

    "${OLMO3_PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import transformers
import vllm

minimum = Version("4.57.0")
installed = Version(transformers.__version__)
if installed < minimum:
    raise SystemExit(
        f"Transformers {installed} does not support OLMo3; version >= {minimum} is required."
    )

from transformers.models.olmo3 import Olmo3Config, Olmo3ForCausalLM  # noqa: F401
from vllm.model_executor.models.registry import ModelRegistry

if "Olmo3ForCausalLM" not in ModelRegistry.get_supported_archs():
    raise SystemExit(f"vLLM {vllm.__version__} does not register Olmo3ForCausalLM")

print(
    f"[OLMo3 preflight] transformers={transformers.__version__} "
    f"vllm={vllm.__version__} architecture=Olmo3ForCausalLM"
)
PY
}
