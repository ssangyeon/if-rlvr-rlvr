#!/usr/bin/env bash
# Sequential runner for the Stage-1 subset ablations, in the agreed priority order:
#   a0 constraint-only -> a3 strict baseline -> a2 soft floor -> a4 penalty -0.1
#   -> a5 no floor -> a1 flip-abstain
#
# Per arm: launch, wait, verify completion (latest checkpoint == 16*TOTAL_EPOCHS),
# reap FSDP shards keeping every epoch's actor/huggingface export for the offline
# eval battery, then tear the GPUs down before the next arm. A failed arm stops
# the chain (the base launcher already auto-resumes 10x internally, so a hard
# failure means something a follow-up run would inherit).
#
# Usage:
#   WANDB_API_KEY=... nohup bash run_all_arms.sh > run_all_arms.log 2>&1 &
# Options:
#   ARMS="a0 a3 a2 a4 a5 a1"   subset/order override
#   TOTAL_EPOCHS=4             epochs per arm (16 steps each)
#   KEEP_FINAL_FULL=0          set 1 to keep the final step's full FSDP checkpoint
#   RUN_TIMEOUT_H=12           per-arm wall-clock cap
set -uo pipefail

SUBSET4K_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-$(cd -- "${SUBSET4K_DIR}/../../.." && pwd)}
[ -d /root/miniforge3/envs/verl/bin ] && export PATH="/root/miniforge3/envs/verl/bin:${PATH}"

ARMS=${ARMS:-"a0 a3 a2 a4 a5 a1"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
KEEP_FINAL_FULL=${KEEP_FINAL_FULL:-0}
RUN_TIMEOUT_H=${RUN_TIMEOUT_H:-12}
EXPECTED_STEPS=$((16 * TOTAL_EPOCHS))
MIN_FREE_GB=220

V2_SUBSET="${VERL_DIR}/.cache/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json"
V1_SUBSET="${VERL_DIR}/.cache/if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.SUBSET4096.json"
LOG_DIR="${VERL_DIR}/logs/subset4k"
mkdir -p "${LOG_DIR}"

declare -A ARM_SCRIPT=(
    [a0]=a0_constraint_only.sh
    [a3]=a3_anchor_strict.sh
    [a2]=a2_anchor_softfloor.sh
    [a4]=a4_anchor_penalty01.sh
    [a5]=a5_anchor_nofloor.sh
    [a1]=a1_anchor_flipabstain.sh
    [b1]=b1_floor_margin.sh
    [b2]=b2_k3min_floor.sh
    [b3]=b3_flipabstain_penalty01.sh
)

say() { echo "[run_all $(date -u +'%m-%d %H:%M:%S')] $*"; }
fatal() { say "FATAL: $*"; exit 1; }

free_gb() { df --output=avail -BG "${VERL_DIR}" | tail -1 | tr -dc '0-9'; }

gpus_clear() {
    local used
    while read -r used; do
        (( used > 1024 )) && return 1
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    return 0
}

teardown() {
    say "teardown: stopping ray + lingering trainers"
    pkill -f "verl.trainer.main_ppo" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
    local waited=0
    until gpus_clear; do
        sleep 15; waited=$((waited + 15))
        if (( waited == 120 )); then
            pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null || true
            pkill -9 -f "ray::" 2>/dev/null || true
            ray stop --force >/dev/null 2>&1 || true
        fi
        (( waited > 600 )) && return 1
    done
    return 0
}

reap_fsdp() {  # $1 = checkpoint dir; keeps actor/huggingface, drops FSDP shards
    local ck=$1 final="global_step_${EXPECTED_STEPS}"
    for d in "${ck}"/global_step_*; do
        [ -d "$d" ] || continue
        if [[ "${KEEP_FINAL_FULL}" == "1" && "$(basename "$d")" == "${final}" ]]; then
            say "  keeping full final checkpoint $(basename "$d")"
            continue
        fi
        if [ ! -d "$d/actor/huggingface" ] || [ -z "$(ls -A "$d/actor/huggingface" 2>/dev/null)" ]; then
            say "  WARNING: $(basename "$d") has no hf export; keeping full checkpoint"
            continue
        fi
        find "$d/actor" -maxdepth 1 -type f \( -name '*.pt' -o -name '*.bin' \) -delete
        say "  reaped FSDP shards in $(basename "$d") (kept actor/huggingface)"
    done
}

# ---- global preflight ----
[ -n "${WANDB_API_KEY:-}" ] || fatal "WANDB_API_KEY missing from environment"
pgrep -f "verl.trainer.main_ppo" >/dev/null && fatal "a trainer is already running on this box"
gpus_clear || fatal "GPUs are not clear; refusing to start"

cd "${VERL_DIR}"
say "arms: ${ARMS} | epochs/arm: ${TOTAL_EPOCHS} (${EXPECTED_STEPS} steps) | timeout ${RUN_TIMEOUT_H}h/arm"

for arm in ${ARMS}; do
    script=${ARM_SCRIPT[$arm]:-}
    [ -n "${script}" ] || fatal "unknown arm: ${arm}"
    tag=$(grep -oE '^ARM_TAG=\S+' "${SUBSET4K_DIR}/${script}" | cut -d= -f2)
    exp="qwen3_4b_sub4k_${tag}_b256_ep${TOTAL_EPOCHS}"
    ck="${VERL_DIR}/checkpoints/verl_if_rlvr/${exp}"
    log="${LOG_DIR}/$(date -u +%m%d_%H%M)_${arm}.log"

    # subset-cache gating: anchor arms need the v2 subset; a0 may fall back to v1
    unset SUBSET_CACHE || true
    if [ ! -s "${V2_SUBSET}" ]; then
        if [ "${arm}" = "a0" ] && [ -s "${V1_SUBSET}" ]; then
            export SUBSET_CACHE="${V1_SUBSET}"
            say "${arm}: v2 subset missing; using v1 subset as row filter (anchors unused in a0)"
        else
            fatal "${arm} needs the v2 subset cache (${V2_SUBSET}); carve it first"
        fi
    fi

    if [ -s "${ck}/latest_checkpointed_iteration.txt" ] && \
       [ "$(cat "${ck}/latest_checkpointed_iteration.txt")" = "${EXPECTED_STEPS}" ]; then
        say "${arm}: already complete (${exp}); skipping"
        continue
    fi

    (( $(free_gb) >= MIN_FREE_GB )) || fatal "only $(free_gb)GB free (< ${MIN_FREE_GB}GB); reap old checkpoints first"

    say "=== ${arm} launching: ${exp} (log: ${log}) ==="
    TOTAL_EPOCHS=${TOTAL_EPOCHS} timeout "$((RUN_TIMEOUT_H * 3600))" \
        bash "${SUBSET4K_DIR}/${script}" > "${log}" 2>&1
    rc=$?

    latest=$(cat "${ck}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "MISSING")
    if [ "${latest}" = "${EXPECTED_STEPS}" ]; then
        say "=== ${arm} COMPLETE (rc=${rc}, latest=${latest}) ==="
        reap_fsdp "${ck}"
    else
        say "=== ${arm} FAILED (rc=${rc}, latest=${latest}, expected ${EXPECTED_STEPS}) — stopping the chain ==="
        say "log tail:"; tail -30 "${log}" || true
        teardown || say "WARNING: GPUs still dirty after failed ${arm}"
        exit 1
    fi

    teardown || fatal "GPUs still dirty after ${arm}; not launching the next arm"
    say "disk: $(free_gb)GB free"
done

say "ALL ARMS COMPLETE"
for arm in ${ARMS}; do
    tag=$(grep -oE '^ARM_TAG=\S+' "${SUBSET4K_DIR}/${ARM_SCRIPT[$arm]}" | cut -d= -f2)
    say "  ${arm}: checkpoints/verl_if_rlvr/qwen3_4b_sub4k_${tag}_b256_ep${TOTAL_EPOCHS}"
done
