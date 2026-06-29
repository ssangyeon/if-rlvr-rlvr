#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/NHNHOME/26msit001_A/IFIF
VERL_DIR=${VERL_DIR:-${ROOT_DIR}/if-rlvr}
TARGET=${TARGET:-${VERL_DIR}/if_rlvr/exps/bidirectional/qwen3_17b_t17b_anchor_pyx005_nonreason.sh}
LOG_DIR=${LOG_DIR:-${ROOT_DIR}/restart_logs}
GPU_SET=${GPU_SET:-4,5,6,7}
POLL_SECONDS=${POLL_SECONDS:-180}
STABLE_EMPTY_CHECKS=${STABLE_EMPTY_CHECKS:-2}
WAIT_PIDS=${WAIT_PIDS:-889450}
POST_WAIT_SECONDS=${POST_WAIT_SECONDS:-180}

mkdir -p "${LOG_DIR}"

echo "[$(date '+%F %T')] waiting before launching ${TARGET}"

if [[ -n "${WAIT_PIDS}" ]]; then
    while true; do
        alive=""
        for pid in ${WAIT_PIDS//,/ }; do
            [[ -z "${pid}" ]] && continue
            if kill -0 "${pid}" 2>/dev/null; then
                alive+="${pid} "
            fi
        done

        if [[ -z "${alive}" ]]; then
            echo "[$(date '+%F %T')] watched PID(s) finished: ${WAIT_PIDS}"
            break
        fi

        echo "[$(date '+%F %T')] waiting for PID(s): ${alive}"
        sleep "${POLL_SECONDS}"
    done
else
    empty_checks=0
    while true; do
        busy=""
        IFS=', ' read -ra gpus <<< "${GPU_SET}"
        for gpu in "${gpus[@]}"; do
            gpu=${gpu// /}
            [[ -z "${gpu}" ]] && continue
            pids=$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
            while IFS= read -r pid; do
                pid=${pid// /}
                [[ -z "${pid}" ]] && continue
                busy+="${gpu}:${pid} "
            done <<< "${pids}"
        done

        if [[ -z "${busy}" ]]; then
            empty_checks=$((empty_checks + 1))
            echo "[$(date '+%F %T')] GPUs ${GPU_SET} empty check ${empty_checks}/${STABLE_EMPTY_CHECKS}"
            if (( empty_checks >= STABLE_EMPTY_CHECKS )); then
                break
            fi
        else
            empty_checks=0
            echo "[$(date '+%F %T')] still busy on GPUs ${GPU_SET}: ${busy}"
        fi

        sleep "${POLL_SECONDS}"
    done
fi

if (( POST_WAIT_SECONDS > 0 )); then
    echo "[$(date '+%F %T')] waiting ${POST_WAIT_SECONDS}s for cleanup before launch"
    sleep "${POST_WAIT_SECONDS}"
fi

source "${ROOT_DIR}/.miniforge3/etc/profile.d/conda.sh"
conda activate verl

cd "${VERL_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export GPU_SET
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export VERL_DIR
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
export RESUME_MODE=${RESUME_MODE:-auto}
export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}

run_log="${LOG_DIR}/qwen3_17b_pyx005_anchor_waitrun_$(date '+%Y%m%d_%H%M%S').log"
echo "[$(date '+%F %T')] launching ${TARGET}; log=${run_log}"
exec bash "${TARGET}" 2>&1 | tee "${run_log}"
