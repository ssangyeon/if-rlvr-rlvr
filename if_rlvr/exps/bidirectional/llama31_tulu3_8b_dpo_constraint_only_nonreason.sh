#!/usr/bin/env bash
# GRPO | allenai/Llama-3.1-Tulu-3-8B-DPO (non-reasoning) | constraint-only IF reward.
#
# No anchor cache, no PPL reward shaping, no LLM judge - the only signal is the
# rule-based IFEval constraint score. Tulu 3 counterpart of
# llama31_8b_constraint_only_nonreason.sh (meta-llama/Llama-3.1-8B-Instruct),
# sized for one node of 8x H100 80GB. Mirrors the reward-disable block used by
# llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh so the
# only difference between that script and this one is the LLM-verifier bonus.
#
# Usage: bash llama31_tulu3_8b_dpo_constraint_only_nonreason.sh
# Any extra arguments are forwarded verbatim as Hydra overrides.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_llama31_tulu3_8b_common.sh
source "${SCRIPT_DIR}/_llama31_tulu3_8b_common.sh"
tulu3_require_runtime

export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_8b_dpo_constraint_only_slot${RUN_SLOT}}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
# 98304/98304/0.8 (this script's previous defaults, inherited from the shared
# base script) deterministically OOM'd in update_actor - "Tried to allocate
# 20.54 GiB" against ~1-2GB free - on all 10 auto-resume attempts on
# 2026-08-20, since this leaves materially less headroom than the values the
# original constraint-only run (wandb 1gyg97hj) actually used. Reverted to
# those proven-stable values. Unlike the gpu_memory_utilization/max_model_len
# differences noted elsewhere as "non-semantic" (true for sampling
# correctness), a too-large token budget is not harmless if it doesn't fit in
# memory at all - this is a capacity fix, not a cosmetic match.
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-49152}
# NOT a ":-" default: _llama31_tulu3_8b_common.sh (sourced above, line 18)
# already runs `export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}`
# itself, before this line ever executes - so by the time a ":-" pattern runs
# here, the variable is already non-empty (0.8) and the fallback never fires.
# This is exactly why the first two attempts at this fix (0.75, then 0.72,
# both written with ":-") silently had no effect and the live process kept
# using 0.8 regardless - confirmed via the running process's own resolved
# args, not assumed. Unconditional assignment is required to actually win.
export ROLLOUT_GPU_MEM_UTIL=0.72
export ROLLOUT_N=${ROLLOUT_N:-8}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-32}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
# Full train split (~94.9k rows) / 1024 = 91 steps per epoch -> ~1 ckpt/epoch,
# matching the original constraint-only run's own checkpoint cadence.
export SAVE_FREQ=${SAVE_FREQ:-91}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}

# No anchor/PPL shaping and no LLM verifier in this run - the reward is purely
# the rule-based IFEval constraint score from the default custom_reward_function.
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.0}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-false}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_PPL_BASELINE=${IF_REF_PPL_BASELINE:-0}
export IF_REF_PPL_ANCHOR=${IF_REF_PPL_ANCHOR:-0}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_constraint_only_b1024_c1}

# The original 4-epoch constraint-only run (wandb 1gyg97hj) set data.seed=42;
# the shared base script below never sets data.seed at all, so it defaults to
# None (an unseeded torch.Generator - see verl/trainer/main_ppo.py's
# create_rl_sampler). Match the original's starting seed here. Note this is a
# partial match, not a full fix: the RandomSampler's generator is advanced by
# one draw per epoch it has already run, so a *fresh* trainer seeded with 42
# reproduces the ORIGINAL run's epoch-1 permutation, not the 5th draw a truly
# continuous 6-epoch run would be on by epoch 5 - that would additionally
# require fast-forwarding the generator through 4 dummy epoch draws, which
# needs the actual per-epoch row count to have stayed constant and isn't
# implemented here. No local dataloader/sampler-state checkpoint survived
# from epochs 1-4 for any of the 4 variants, so exact epoch-5 shuffle-order
# reproduction isn't achievable regardless - this at least removes the
# additional, unnecessary seed-value mismatch on top of that. The other 3
# variants' original runs all used data.seed=None (unseeded) too, so their
# continuations already match exactly with no override needed.
export DATA_SEED=${DATA_SEED:-42}

exec bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" \
    "data.seed=${DATA_SEED}" \
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
