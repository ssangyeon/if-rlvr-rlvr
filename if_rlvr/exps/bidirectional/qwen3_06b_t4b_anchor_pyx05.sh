#!/usr/bin/env bash
# Qwen3-0.6B reasoning policy with 4B non-reasoning final-answer anchors,
# rescored by the Qwen3-0.6B reference policy. Uses cached-anchor rows only
# so train_batch_size=512 and total_epochs=3 give 183 * 3 = 549 steps.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-/NHNHOME/26msit001_A/IFIF/if-rlvr}
CACHE_ROOT=${CACHE_ROOT:-/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr/.cache}

cd "${VERL_DIR}"

export GPU_SET=${GPU_SET:-4,5,6,7}
export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-0.6B}
export ENABLE_THINKING=${ENABLE_THINKING:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-true}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.5}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}

export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher4b_nonreason_train_seed1_val512_scored_by_qwen3_06b.json}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-true}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}

if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "Missing anchor cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    echo "Run precompute_teacher4b_anchor_scored_by_qwen3_06b.sh first." >&2
    exit 1
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_06b_grpo_think_pyx05_t4banchor_s06b_b512_c1}

exec bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh"
