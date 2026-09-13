#!/usr/bin/env bash
# Qwen3-1.7B non-reasoning baseline: constraint_reward + p_ref(final_answer | x).
# p_ref = exp(-summed NLL), with coefficient 1.0 by default. The extracted final
# answer is scored under x only, using the same tokens as the inverse-PPL baseline.
# There are no anchors, length normalization, or IF gate.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
export VERL_DIR
cd "${VERL_DIR}"

export GPU_SET=${GPU_SET:-4,5,6,7}
export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
export SAVE_FREQ=${SAVE_FREQ:-91}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

export IF_PPL_REWARD_STRATEGY=raw_likelihood
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-1.0}
export PX_GIVEN_Y_REWARD_COEFF=0.0
# Reuse the frozen-reference worker and x-only prefix construction.
export IF_REF_POLICY_ANCHOR_PPL=true
export IF_REF_ANCHOR_PRECOMPUTE=false
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=false
export IF_REF_PPL_GATE=false
export IF_REF_PPL_BASELINE=0
export IF_REF_PPL_ANCHOR=0

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_17b_grpo_nonthink_raw_likelihood_coeff${PY_GIVEN_X_REWARD_COEFF}_b${TRAIN_BATCH_SIZE}_c1}

exec bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" "$@"
