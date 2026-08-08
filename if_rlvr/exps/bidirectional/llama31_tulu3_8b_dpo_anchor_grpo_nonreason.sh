#!/usr/bin/env bash
# Four-epoch GRPO for allenai/Llama-3.1-Tulu-3-8B-DPO using its validated
# one-sample x/x+c anchor cache. Tulu/paper-specific: temp=1, response=2048,
# actor LR=5e-7. Current comparison setup retained: batch=1024, rollout n=8.

set -euo pipefail

export GPU_SET=${GPU_SET:-4,5,6,7}
export RUN_SLOT=${RUN_SLOT:-1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_llama31_tulu3_8b_common.sh
source "${SCRIPT_DIR}/_llama31_tulu3_8b_common.sh"
tulu3_require_runtime

export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_8b_dpo_anchor_grpo_slot${RUN_SLOT}}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-98304}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-98304}
export ROLLOUT_N=${ROLLOUT_N:-8}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-32}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
export SAVE_FREQ=${SAVE_FREQ:-91}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}

export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-true}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}

if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "Downloading validated Tulu anchor cache from hf://datasets/${TULU3_ANCHOR_HF_REPO}/${TULU3_ANCHOR_CACHE_FILENAME}"
    "${TULU3_PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    os.environ["TULU3_ANCHOR_HF_REPO"],
    filename=os.environ["TULU3_ANCHOR_CACHE_FILENAME"],
    repo_type="dataset",
    local_dir=os.environ["CACHE_ROOT"],
)
print(f"Downloaded anchor cache: {path}")
PY
fi
if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "Missing Tulu anchor cache after download: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    echo "Run precompute_teacher_llama31_tulu3_8b_dpo_anchor_scored_by_llama31_tulu3_8b.sh first." >&2
    exit 1
fi

if [[ "${TULU3_VALIDATE_ANCHOR_BEFORE_TRAIN:-true}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    "${TULU3_PYTHON_BIN}" "${SCRIPT_DIR}/validate_upload_tulu3_anchor_cache.py" \
        --no-upload \
        --cache "${IF_REF_ANCHOR_CACHE_PATH}" \
        --report "${CACHE_ROOT}/${TULU3_ANCHOR_VALIDATION_FILENAME}" \
        --filename "${TULU3_ANCHOR_CACHE_FILENAME}" \
        --model "${MODEL_PATH}" \
        --model-revision "${TULU3_MODEL_REVISION}" \
        --dataset-revision "${TULU3_IF_DATASET_REVISION}" \
        --max-prompt-length "${MAX_PROMPT_LENGTH}" \
        --max-response-length "${MAX_RESPONSE_LENGTH}" \
        --temperature "${ROLLOUT_TEMPERATURE}" \
        --top-p "${ROLLOUT_TOP_P}"
fi

PY_GIVEN_X_REWARD_TAG=${PY_GIVEN_X_REWARD_COEFF/./}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_anchor_pyx${PY_GIVEN_X_REWARD_TAG}_b1024_c1_t1_2k}

exec bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" \
    +if_ppl_prefix_mode=standard \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_APPLY_ENABLE_THINKING_KWARG=\"${IF_APPLY_ENABLE_THINKING_KWARG}\"" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_ALLOW_MISSING_THINK_FINAL_ANSWER=\"${IF_ALLOW_MISSING_THINK_FINAL_ANSWER}\"" \
    "$@"
