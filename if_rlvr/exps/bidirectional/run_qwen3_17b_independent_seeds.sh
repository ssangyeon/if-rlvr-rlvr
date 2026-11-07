#!/usr/bin/env bash
# Five independent 4-epoch RL runs; fixed split and fixed precomputed anchors.
# Usage: VERL_DIR=/path/to/if-rlvr CACHE_ROOT=/path/to/.cache bash this.sh
# Preview: DRY_RUN=1 VERL_DIR=/path/to/if-rlvr bash this.sh
# Select seeds: TRAIN_SEEDS="1 2 3 4 5"; run one: TRAIN_SEED=3.
set -euo pipefail
if [[ -z "${TRAIN_SEED:-}" ]]; then
    read -r -a seeds <<< "${TRAIN_SEEDS:-1 2 3 4 5}"
    [[ ${#seeds[@]} -gt 0 ]] || exit 1
    for run_seed in "${seeds[@]}"; do
        TRAIN_SEED="$run_seed" bash "$0"
    done
    exit 0
fi
[[ "$TRAIN_SEED" =~ ^[1-9][0-9]*$ ]] && (( TRAIN_SEED < 2147483647 )) || { echo "Invalid TRAIN_SEED" >&2; exit 1; }
: "${VERL_DIR:?Set VERL_DIR to the training repository root}"
SCRIPT_DIR="${VERL_DIR}/if_rlvr/exps/bidirectional"
export IF_DATA_SEED=1
export IF_VAL_SIZE=512
export RESUME_MODE=disable
export IF_MAX_RETRIES=1
export PYTHONHASHSEED="$TRAIN_SEED"
export EXPERIMENT_NAME="${EXPERIMENT_PREFIX:-qwen3_17b_grpo_nonthink_pyx01_t17banchor_s17b_b1024_c1}_seed${TRAIN_SEED}"
export IF_RLVR_RUN_ID="$EXPERIMENT_NAME"
# Clear inherited tracking IDs so each training run creates its own W&B run.
unset WANDB_RUN_ID WANDB_RESUME
checkpoint_dir="${CHECKPOINT_ROOT:-${VERL_DIR}/checkpoints/independent_rl}/${EXPERIMENT_NAME}"
if [[ -e "$checkpoint_dir" && "${DRY_RUN:-0}" != 1 ]]; then
    echo "Checkpoint directory already exists: $checkpoint_dir. Choose a new EXPERIMENT_PREFIX." >&2
    exit 1
fi
CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}

if [[ "${DRY_RUN:-0}" != 1 ]]; then cd "${VERL_DIR}"; fi

export GPU_SET=${GPU_SET:-4,5,6,7}
export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-1.7B}
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
export SAVE_FREQ=${SAVE_FREQ:-91}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}

export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher17b_nonreason_train_seed1_val512_scored_by_qwen3_17b.json}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}

if [[ "${DRY_RUN:-0}" != 1 && ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "Missing anchor cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    echo "Run precompute_teacher17b_anchor_scored_by_qwen3_17b.sh first." >&2
    exit 1
fi

# Explicit vLLM engine seed is supported by this repository's rollout config.
# No full_determinism toggle: preserve the original training kernels/settings.
cmd=(bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh"
    "data.seed=${TRAIN_SEED}"
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${TRAIN_SEED}"
    "trainer.default_local_dir=${checkpoint_dir}"
    "trainer.resume_mode=disable")
printf 'Run: %s | split seed: 1 | RL seed: %s | epochs: %s\n' "$EXPERIMENT_NAME" "$TRAIN_SEED" "$TOTAL_EPOCHS"
printf 'Anchor: %s\n' "$IF_REF_ANCHOR_CACHE_PATH"
printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
exec "${cmd[@]}"
