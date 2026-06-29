#!/usr/bin/env bash
# Resume a Qwen3-0.6B ref-anchor reasoning run using the precomputed anchor cache.

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

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-0.6B}
export ENABLE_THINKING=${ENABLE_THINKING:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-true}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.5}

# Reuse the precomputed Qwen3-0.6B reasoning anchor cache by default.
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_qwen3_06b_const1_train_seed1_val512_thinktrue_fresh_20260615_163405.json}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-true}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-false}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_06b_if_grpo_vllm_fsdp_think_true_pyx_anchor_0.5_pxy_0.0_clip_use_bsz_512_const1only_refanchor_cached_refpolicy_qwen3_06b}

exec bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh"
