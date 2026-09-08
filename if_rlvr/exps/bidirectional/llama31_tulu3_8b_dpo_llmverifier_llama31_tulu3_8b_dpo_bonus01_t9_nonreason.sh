#!/usr/bin/env bash
# GRPO | allenai/Llama-3.1-Tulu-3-8B-DPO (non-reasoning)
#   + the same Tulu-3-8B-DPO model as the LLM verifier
#   +0.1 verifier bonus at score >= 9
#   no anchor/PPL reward shaping

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_LAUNCHER="${SCRIPT_DIR}/llama31_tulu3_8b_dpo_llmverifier_llama31_tulu3_8b_dpo_bonus01_t7_nonreason.sh"

[[ -f "${BASE_LAUNCHER}" ]] || {
    echo "ERROR: missing base launcher: ${BASE_LAUNCHER}" >&2
    exit 1
}

# Override every threshold-bearing identity so this is a clean experiment and
# cannot resume from or write into the threshold-7 checkpoint directory.
export IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-9}
export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-${GPU_SET}}
export RUN_SLOT=${RUN_SLOT:-4}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_8b_dpo_tulu3_8b_dpo_t9_gpu0123_slot4}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-28000}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_llmverifier_llama31_tulu3_8b_dpo_nonthink_bonus01_threshold9_b1024_c1_t1_2k}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-6}
export HF_CHECKPOINT_REPO_NAME=${HF_CHECKPOINT_REPO_NAME:-tulu3-8b-dpo-grpo-tulu3-8b-dpo-t9}

exec bash "${BASE_LAUNCHER}" "$@"
