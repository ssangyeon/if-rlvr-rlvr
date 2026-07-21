#!/usr/bin/env bash
# 95% CI inner interval cache:
# lower threshold = upper CI endpoint, upper threshold = lower CI endpoint.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
WORKSPACE_DIR=$(cd -- "${VERL_DIR}/.." && pwd)
export VERL_ENV_BIN=${VERL_ENV_BIN:-${WORKSPACE_DIR}/.miniforge3/envs/verl/bin}
export PATH="${VERL_ENV_BIN}:${PATH}"
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${WORKSPACE_DIR}/IFBench/.nltk_data}
export NLTK_DATA=${NLTK_DATA:-${NLTK_DATA_DIR}}

if [[ ! -x "${VERL_ENV_BIN}/python3" ]]; then
    echo "verl Python not found: ${VERL_ENV_BIN}/python3" >&2
    exit 1
fi

if [[ ! -d "${NLTK_DATA_DIR}/tokenizers/punkt_tab/english" ]]; then
    echo "NLTK punkt_tab is missing: ${NLTK_DATA_DIR}/tokenizers/punkt_tab/english" >&2
    exit 1
fi

export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-0}
export IF_DATA_SEED=${IF_DATA_SEED:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}

export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher17b_nonreason_train_seed1_val512_scored_by_qwen3_17b_ci95_inner_s1to8.json}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}

if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "Missing CI95 inner anchor cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    echo "Run build_teacher17b_ci95_anchor_caches.py first." >&2
    exit 1
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_17b_grpo_nonthink_pyx01_t17banchor_ci95inner_s1to8_s17b_b1024_c1}

exec bash "${SCRIPT_DIR}/qwen3_17b_t17b_anchor_pyx01_nonreason.sh" "$@"
