#!/usr/bin/env bash
# Tulu-3-8B-DPO policy + fixed Tulu-3-8B-DPO IntentCheck judge, non-reasoning.
# Replace ONLY the extra-bonus evaluator, not the constraint reward:
#   C=0 -> 0 (skip judge); C>0 + YES -> C+0.1; C>0 + NO/error -> C.
# IntentCheck prompt: arXiv:2508.04632v2, pp. 27-28.
# The paper's full AND/zero reward is deliberately NOT used here.
# --dry-run prints the new settings without starting servers or training.

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_LAUNCHER="${SCRIPT_DIR}/llama31_tulu3_8b_dpo_llmverifier_llama31_tulu3_8b_dpo_bonus01_t9_nonreason.sh"
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}

export MODEL_PATH=${MODEL_PATH:-allenai/Llama-3.1-Tulu-3-8B-DPO}
export TULU3_MODEL_REVISION=${TULU3_MODEL_REVISION:-a7beb67e33ffd01cc87ac3b46cadc1000985b8db}
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-${MODEL_PATH}}
export IF_LLM_VERIFIER_REVISION=${IF_LLM_VERIFIER_REVISION:-${TULU3_MODEL_REVISION}}
[[ "${IF_LLM_VERIFIER_MODEL}" == "${MODEL_PATH}" && "${IF_LLM_VERIFIER_REVISION}" == "${TULU3_MODEL_REVISION}" ]] || {
    echo "ERROR: this experiment requires the same initial Tulu policy/judge checkpoint." >&2
    exit 1
}
export ENABLE_THINKING=false
export IF_LLM_VERIFIER_ENABLE_THINKING=false
export IF_REQUIRE_THINK_END_FOR_REWARD=false
export IF_LLM_VERIFIER_MODE=intentcheck
export IF_LLM_VERIFIER_RESPONSE_FORMAT=false
export IF_LLM_VERIFIER_BONUS=0.1
# Retain the existing Tulu judge decoding defaults, including its token budget.
# Non-reasoning does not prohibit the checklist requested in the judge's answer.
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-1024}

export PY_GIVEN_X_REWARD_COEFF=0.0
export PX_GIVEN_Y_REWARD_COEFF=0.0
export IF_REF_ANCHOR_PRECOMPUTE=false
export IF_REF_POLICY_ANCHOR_PPL=false
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=false
export IF_LLM_VERIFIER_ANCHOR_FALLBACK_ONLY=false

export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export SAVE_FREQ=${SAVE_FREQ:-91}
# A fresh run must not silently restart from zero after a crash.
export IF_MAX_RETRIES=${IF_MAX_RETRIES:-1}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-${GPU_SET}}
export RUN_SLOT=${RUN_SLOT:-5}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-30000}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_self_intentcheck_$(date +%Y%m%d_%H%M%S)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_intentcheck_tulu3_8b_dpo_bonus01_${IF_RLVR_RUN_ID}}
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export CKPT_DIR=${CKPT_DIR:-${VERL_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier/${IF_RLVR_RUN_ID}}
export WANDB_RUN_ID=${WANDB_RUN_ID:-ti$(date +%Y%m%d%H%M%S)}
export WANDB_RESUME=never
export HF_CHECKPOINT_PUSH=false
export HF_CHECKPOINT_REPO_NAME=${HF_CHECKPOINT_REPO_NAME:-tulu3-8b-dpo-grpo-self-intentcheck}

# Explicit reward kwargs propagate through Ray even if custom env vars do not.
# Keep 'false' a STRING: the legacy manager treats a Hydra boolean False as unset.
INTENT_OVERRIDES=(
    "++reward.reward_kwargs.if_llm_verifier_mode=intentcheck"
    "++reward.reward_kwargs.if_llm_verifier_enable_thinking='false'"
    "++reward.reward_kwargs.if_llm_verifier_response_format=false"
    "++reward.reward_kwargs.if_llm_verifier_anchor_fallback_only=false"
    "++reward.reward_kwargs.verification_reward=1.0"
    "trainer.resume_mode=disable"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%s\n' \
        "policy=${MODEL_PATH}@${TULU3_MODEL_REVISION}" \
        "judge=${IF_LLM_VERIFIER_MODEL}@${IF_LLM_VERIFIER_REVISION}" \
        "mode=intentcheck; numeric threshold=unused; bonus=0.1" \
        "policy/judge non-reasoning; judge max_tokens=${IF_LLM_VERIFIER_MAX_TOKENS}" \
        "reward: C=0 -> 0/skip; C>0 and YES -> C+0.1; NO/error -> C" \
        "GPU_SET=${GPU_SET}; anchor/PPL=off; resume=disable" \
        "total_epochs=${TOTAL_EPOCHS}; save_freq=${SAVE_FREQ}" \
        "experiment=${EXPERIMENT_NAME}" \
        "checkpoints=${CKPT_DIR}" \
        "verifier_logs=${IF_LLM_VERIFIER_LOG_DIR}" \
        "Hydra overrides:" "${INTENT_OVERRIDES[@]}"
    exit 0
fi

[[ -f "${BASE_LAUNCHER}" ]] || { echo "ERROR: missing ${BASE_LAUNCHER}" >&2; exit 1; }
echo "[IntentCheck] YES/NO bonus evaluator; the parent launcher's numeric threshold is NOT used."
exec bash "${BASE_LAUNCHER}" "${INTENT_OVERRIDES[@]}" "$@"
