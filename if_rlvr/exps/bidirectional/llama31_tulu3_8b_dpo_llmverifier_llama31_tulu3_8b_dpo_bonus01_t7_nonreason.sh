#!/usr/bin/env bash
# GRPO | allenai/Llama-3.1-Tulu-3-8B-DPO (non-reasoning)
#   + the same Tulu-3-8B-DPO model as the LLM verifier
#   +0.1 verifier bonus at score >= 7
#   no anchor/PPL reward shaping
#
# This is intentionally a thin, auditable specialization of
# llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh.
# The parent launcher owns the reward manager and server lifecycle; this file
# changes only the verifier identity and current-host placement defaults.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARENT_LAUNCHER="${SCRIPT_DIR}/llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh"

[[ -f "${PARENT_LAUNCHER}" ]] || {
    echo "ERROR: missing parent launcher: ${PARENT_LAUNCHER}" >&2
    exit 1
}

########################### exact experiment definition ###########################
export MODEL_PATH=${MODEL_PATH:-allenai/Llama-3.1-Tulu-3-8B-DPO}
export TULU3_MODEL_REVISION=${TULU3_MODEL_REVISION:-a7beb67e33ffd01cc87ac3b46cadc1000985b8db}
export ENABLE_THINKING=${ENABLE_THINKING:-false}

export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-${MODEL_PATH}}
export IF_LLM_VERIFIER_REVISION=${IF_LLM_VERIFIER_REVISION:-${TULU3_MODEL_REVISION}}
export IF_LLM_VERIFIER_ENABLE_THINKING=${IF_LLM_VERIFIER_ENABLE_THINKING:-false}
export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-7}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}

# Preserve the source launcher's four-way judge throughput: one TP1 Tulu
# verifier per training GPU.  On this host each B200 has 183 GiB, so the 8B
# verifier can remain resident beside the trainer.  This keeps the exact same
# reward calculation while avoiding four 14.958-GiB CPU sleep backups.
export GPU_SET=${GPU_SET:-4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-${GPU_SET}}
export IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP:-1}
export IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE:-false}
export IF_LLM_VERIFIER_MANAGE_SLEEP=${IF_LLM_VERIFIER_MANAGE_SLEEP:-false}
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.15}
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-8192}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-32}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}

# Keep the actor/optimizer on the 183-GiB B200s.  Host offload would compete
# with the verifier's sleep backup under the container cgroup.
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
export IF_RLVR_OBJECT_STORE_BYTES=${IF_RLVR_OBJECT_STORE_BYTES:-2147483648}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-4}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-16}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-16}

# Isolate this fresh run from the failed resume attempt's Ray/port namespace.
export RUN_SLOT=${RUN_SLOT:-3}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_8b_dpo_tulu3_8b_dpo_t7_gpu4567_slot3}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-26000}

export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_llmverifier_llama31_tulu3_8b_dpo_nonthink_bonus01_threshold7_b1024_c1_t1_2k}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export SAVE_FREQ=${SAVE_FREQ:-92}
export HF_CHECKPOINT_PUSH=${HF_CHECKPOINT_PUSH:-false}
export HF_CHECKPOINT_REPO_NAME=${HF_CHECKPOINT_REPO_NAME:-tulu3-8b-dpo-grpo-tulu3-8b-dpo-t7}

VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier/${IF_RLVR_RUN_ID}}

########################### destructive-OOM preflight ###########################
# The co-resident profile removes the 59.8-GiB verifier sleep backup, but four
# server processes, Ray, the trainer, and its 4-GiB object store still must
# share the cgroup with any jobs on GPUs 0-3.  Refuse the known-bad 32-GiB
# container rather than risk OOM-killing that unrelated job.
cgroup_rel=$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)
cgroup_min_bytes=""
if [[ -n "${cgroup_rel}" && -d "/sys/fs/cgroup${cgroup_rel}" ]]; then
    cgroup_cursor="/sys/fs/cgroup${cgroup_rel}"
    while [[ "${cgroup_cursor}" == /sys/fs/cgroup* ]]; do
        if [[ -r "${cgroup_cursor}/memory.max" ]]; then
            cgroup_value=$(tr -d '[:space:]' < "${cgroup_cursor}/memory.max")
            if [[ "${cgroup_value}" =~ ^[0-9]+$ ]] && \
               { [[ -z "${cgroup_min_bytes}" ]] || (( cgroup_value < cgroup_min_bytes )); }; then
                cgroup_min_bytes="${cgroup_value}"
            fi
        fi
        [[ "${cgroup_cursor}" == "/sys/fs/cgroup" ]] && break
        cgroup_cursor=$(dirname -- "${cgroup_cursor}")
    done
fi

required_cgroup_bytes=$((64 * 1024 * 1024 * 1024))
if [[ -n "${cgroup_min_bytes}" ]] && (( cgroup_min_bytes < required_cgroup_bytes )); then
    if [[ ! "${IF_RLVR_FORCE_LOW_CGROUP_MEMORY:-false}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "ERROR: effective cgroup memory.max is $((cgroup_min_bytes / 1024 / 1024 / 1024)) GiB." >&2
        echo "       This co-resident Tulu-verifier profile requires at least 64 GiB" >&2
        echo "       while another job shares the cgroup (96 GiB recommended)." >&2
        echo "       Set IF_RLVR_FORCE_LOW_CGROUP_MEMORY=1 only for an explicitly authorized forced attempt." >&2
        exit 1
    fi
    echo "WARNING: forcing launch below the 64-GiB cgroup safety floor." >&2
    echo "         effective memory.max=$((cgroup_min_bytes / 1024 / 1024 / 1024)) GiB" >&2
    # If the shared cgroup does OOM, prefer killing this explicitly forced run
    # over an unrelated process already using GPUs 0-3. Children inherit this.
    if [[ -w /proc/self/oom_score_adj ]]; then
        echo 1000 > /proc/self/oom_score_adj
    fi
fi

gpu_state=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
IFS=',' read -r -a requested_gpus <<< "${GPU_SET//[[:space:]]/}"
for gpu in "${requested_gpus[@]}"; do
    used=$(awk -F, -v target="${gpu}" '$1 + 0 == target {gsub(/[[:space:]]/, "", $2); print $2}' <<<"${gpu_state}")
    total=$(awk -F, -v target="${gpu}" '$1 + 0 == target {gsub(/[[:space:]]/, "", $3); print $3}' <<<"${gpu_state}")
    [[ -n "${used}" && "${used}" -le 1024 ]] || {
        echo "ERROR: GPU ${gpu} is not free enough (${used:-unknown} MiB used)." >&2
        exit 1
    }
    [[ -n "${total}" && "${total}" -ge 170000 ]] || {
        echo "ERROR: GPU ${gpu} has only ${total:-unknown} MiB; the co-resident profile requires a 180-GiB-class GPU." >&2
        exit 1
    }
done

exec bash "${PARENT_LAUNCHER}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +ray_kwargs.ray_init.num_cpus=16 \
    "$@"
