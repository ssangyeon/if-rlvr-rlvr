#!/usr/bin/env bash
# Shared, non-thinking runtime defaults for allenai/Llama-3.1-Tulu-3-8B-DPO.

if [[ -n "${IF_RLVR_TULU3_COMMON_SH:-}" ]]; then
    return 0
fi
export IF_RLVR_TULU3_COMMON_SH=1

TULU3_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${TULU3_SCRIPT_DIR}/../../.." && pwd)}
export IFIF_ROOT=${IFIF_ROOT:-$(cd -- "${VERL_DIR}/.." && pwd)}
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${IFIF_ROOT}/IFBench/.nltk_data}

# Reuse the tested modern Transformers/vLLM overlay. It supports both OLMo3 and
# the standard LlamaForCausalLM architecture used by Tulu 3.
export TULU3_CONDA_ENV_DIR=${TULU3_CONDA_ENV_DIR:-${IFIF_ROOT}/.miniforge3/envs/verl}
export TULU3_CONDA_SH=${TULU3_CONDA_SH:-${IFIF_ROOT}/.miniforge3/etc/profile.d/conda.sh}
export TULU3_VENV_DIR=${TULU3_VENV_DIR:-${IFIF_ROOT}/.venvs/verl-olmo3}
export TULU3_PYTHON_BIN=${TULU3_PYTHON_BIN:-${TULU3_VENV_DIR}/bin/python}
if [[ "${CONDA_PREFIX:-}" != "${TULU3_CONDA_ENV_DIR}" ]]; then
    if [[ ! -r "${TULU3_CONDA_SH}" ]]; then
        echo "Missing conda activation script: ${TULU3_CONDA_SH}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${TULU3_CONDA_SH}"
    conda activate "${TULU3_CONDA_ENV_DIR}"
fi
if [[ -x "${TULU3_PYTHON_BIN}" ]]; then
    export PATH="$(dirname -- "${TULU3_PYTHON_BIN}"):${PATH}"
fi

export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export MODEL_PATH=${MODEL_PATH:-allenai/Llama-3.1-Tulu-3-8B-DPO}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-${MODEL_PATH}}
export TULU3_MODEL_REVISION=${TULU3_MODEL_REVISION:-a7beb67e33ffd01cc87ac3b46cadc1000985b8db}
export TULU3_IF_DATASET_REVISION=${TULU3_IF_DATASET_REVISION:-2e3a77407b7fce69f95b248d64a884e3ae1c2423}
export TULU3_VERIFY_REMOTE_REVISIONS=${TULU3_VERIFY_REMOTE_REVISIONS:-true}

# Tulu 3 DPO is a non-reasoning post-trained model. Its embedded chat template
# does not use Qwen's enable_thinking kwarg and emits no </think> delimiter.
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-false}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}
export IF_ALLOW_MISSING_THINK_FINAL_ANSWER=${IF_ALLOW_MISSING_THINK_FINAL_ANSWER:-true}
export IF_PPL_PREFIX_MODE=${IF_PPL_PREFIX_MODE:-standard}

export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}
export TULU3_ANCHOR_HF_REPO=${TULU3_ANCHOR_HF_REPO:-sangyon/anchor_cache}
export TULU3_ANCHOR_HF_TOKEN_FILE=${TULU3_ANCHOR_HF_TOKEN_FILE:-/tmp/ifif_hf_tulu_anchor_upload_auth/token}
export TULU3_ANCHOR_CACHE_FILENAME=${TULU3_ANCHOR_CACHE_FILENAME:-if_ref_anchor_teacher_llama31_tulu3_8b_dpo_nonreason_train_seed1_val512_t1_p095_r2048_scored_by_llama31_tulu3_8b_dpo.json}
export TULU3_ANCHOR_VALIDATION_FILENAME=${TULU3_ANCHOR_VALIDATION_FILENAME:-${TULU3_ANCHOR_CACHE_FILENAME%.json}.validation.json}
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/${TULU3_ANCHOR_CACHE_FILENAME}}

# The paper baseline uses temperature 1 and a 2,048-token response cutoff.
# Other rollout settings intentionally retain the current OLMo comparison run.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-40960}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
export PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

tulu3_require_runtime() {
    if [[ ! -x "${TULU3_PYTHON_BIN}" ]]; then
        echo "Missing Tulu runtime: ${TULU3_PYTHON_BIN}" >&2
        return 1
    fi

    "${TULU3_PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import transformers
import vllm
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from vllm.model_executor.models.registry import ModelRegistry

model_id = os.environ["MODEL_PATH"]
model_revision = os.environ["TULU3_MODEL_REVISION"]
dataset_revision = os.environ["TULU3_IF_DATASET_REVISION"]
verify_remote = os.environ.get("TULU3_VERIFY_REMOTE_REVISIONS", "true").lower() in {"1", "true", "yes", "on"}

if "LlamaForCausalLM" not in ModelRegistry.get_supported_archs():
    raise SystemExit(f"vLLM {vllm.__version__} does not register LlamaForCausalLM")

if verify_remote:
    api = HfApi()
    actual_model_revision = api.model_info(model_id).sha
    actual_dataset_revision = api.dataset_info("allenai/IF_multi_constraints_upto5").sha
    if actual_model_revision != model_revision:
        raise SystemExit(
            f"Tulu model revision changed: expected {model_revision}, got {actual_model_revision}"
        )
    if actual_dataset_revision != dataset_revision:
        raise SystemExit(
            f"IF dataset revision changed: expected {dataset_revision}, got {actual_dataset_revision}"
        )

metadata_dir = Path(os.environ["VERL_DIR"]) / ".cache/model_metadata/llama31_tulu3_8b_dpo"
tokenizer_source = str(metadata_dir) if (metadata_dir / "tokenizer_config.json").is_file() else model_id
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_source,
    revision=None if tokenizer_source == str(metadata_dir) else model_revision,
    local_files_only=tokenizer_source == str(metadata_dir),
)
if not tokenizer.chat_template:
    raise SystemExit("Tulu tokenizer has no embedded chat template")
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Runtime preflight."}],
    add_generation_prompt=True,
    tokenize=False,
)
if "<|assistant|>\n" not in rendered or "<think>" in rendered:
    raise SystemExit(f"Unexpected Tulu chat template rendering: {rendered!r}")

print(
    f"[Tulu3 preflight] transformers={transformers.__version__} "
    f"vllm={vllm.__version__} architecture=LlamaForCausalLM "
    f"tokenizer={tokenizer.__class__.__name__}"
)
PY
}
