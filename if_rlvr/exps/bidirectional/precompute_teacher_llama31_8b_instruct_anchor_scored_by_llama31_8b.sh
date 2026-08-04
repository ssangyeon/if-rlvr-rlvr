#!/usr/bin/env bash
# Generate non-reasoning Llama-3.1-8B-Instruct anchors y0/y1, score them with
# the same Llama reference policy, and persist the anchor cache.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_llama31_8b_common.sh
source "${SCRIPT_DIR}/_llama31_8b_common.sh"
llama31_prepare_model_snapshot
llama31_require_runtime

export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-llama31_8b_anchor_precompute_slot${RUN_SLOT}}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-512}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-98304}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-98304}
export ROLLOUT_N=${ROLLOUT_N:-2}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
export IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE=${IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE:-4096}
export IF_REF_ANCHOR_CACHE_SAVE_INTERVAL=${IF_REF_ANCHOR_CACHE_SAVE_INTERVAL:-1024}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-false}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}

export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.5}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_8b_precompute_teacher8b_nonreason_anchor_scored_by_llama31_8b}

# Precompute is restartable from the cache. Do not retry deterministic setup or
# authorization failures inside the generic auto-resume loop.
export IF_MAX_RETRIES=${IF_MAX_RETRIES:-1}

exec bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" \
    +if_ref_anchor_precompute_only=true \
    +if_ref_anchor_cache_save_interval="${IF_REF_ANCHOR_CACHE_SAVE_INTERVAL}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_APPLY_ENABLE_THINKING_KWARG=\"${IF_APPLY_ENABLE_THINKING_KWARG}\"" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_ALLOW_MISSING_THINK_FINAL_ANSWER=\"${IF_ALLOW_MISSING_THINK_FINAL_ANSWER}\"" \
    trainer.logger='["console"]' \
    "$@"
