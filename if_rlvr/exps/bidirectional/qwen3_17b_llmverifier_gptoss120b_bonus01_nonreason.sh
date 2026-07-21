#!/usr/bin/env bash
# Qwen3-1.7B non-reasoning policy with rule IF reward plus an external
# openai/gpt-oss-120b LLM verifier bonus.
#
# The verifier and policy alternate on GPU 0:
#   1. policy rollout on GPUs 0,1,2,3
#   2. policy rollout engines sleep
#   3. verifier wakes one DP replica on each of GPUs 0,1,2,3 and computes reward
#   4. verifier sleeps
#   5. log-prob, KL, and actor update run on GPUs 0,1,2,3

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr}
CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}

cd "${VERL_DIR}"

export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-$((20000 + RUN_SLOT * 2000))}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-$((IF_RLVR_PORT_BASE + 200))}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

########################### verifier server ###########################
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-openai/gpt-oss-120b}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-0,1,2,3}
export IF_LLM_VERIFIER_REPLICA_SERVERS=${IF_LLM_VERIFIER_REPLICA_SERVERS:-true}
export IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP:-1}
export IF_LLM_VERIFIER_DP=${IF_LLM_VERIFIER_DP:-1}
export IF_LLM_VERIFIER_HOST=${IF_LLM_VERIFIER_HOST:-127.0.0.1}
export IF_LLM_VERIFIER_PORT=${IF_LLM_VERIFIER_PORT:-$((IF_RLVR_PORT_BASE + 900))}
export IF_LLM_VERIFIER_PORTS=${IF_LLM_VERIFIER_PORTS:-}
export IF_LLM_VERIFIER_BASE_URL=${IF_LLM_VERIFIER_BASE_URL:-http://${IF_LLM_VERIFIER_HOST}:${IF_LLM_VERIFIER_PORT}/v1}
export IF_LLM_VERIFIER_BASE_URLS=${IF_LLM_VERIFIER_BASE_URLS:-}
export IF_LLM_VERIFIER_START_SERVER=${IF_LLM_VERIFIER_START_SERVER:-true}
export IF_LLM_VERIFIER_PYTHON=${IF_LLM_VERIFIER_PYTHON:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/.miniforge3/envs/verl/bin/python}
export IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE:-true}
export IF_LLM_VERIFIER_MANAGE_SLEEP=${IF_LLM_VERIFIER_MANAGE_SLEEP:-true}
export IF_LLM_VERIFIER_DEV_MODE=${IF_LLM_VERIFIER_DEV_MODE:-1}
export IF_LLM_VERIFIER_SLEEP_LEVEL=${IF_LLM_VERIFIER_SLEEP_LEVEL:-1}
export IF_LLM_VERIFIER_CONTROL_TIMEOUT=${IF_LLM_VERIFIER_CONTROL_TIMEOUT:-300}
export IF_LLM_VERIFIER_WAIT_TIMEOUT=${IF_LLM_VERIFIER_WAIT_TIMEOUT:-900}
export IF_LLM_VERIFIER_DTYPE=${IF_LLM_VERIFIER_DTYPE:-bfloat16}
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.9}
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-24576}
export IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS=${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS:-32768}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-128}
export IF_LLM_VERIFIER_TRUST_REMOTE_CODE=${IF_LLM_VERIFIER_TRUST_REMOTE_CODE:-false}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier}

export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-5}
export IF_LLM_VERIFIER_TEMPERATURE=${IF_LLM_VERIFIER_TEMPERATURE:-0.0}
export IF_LLM_VERIFIER_TOP_P=${IF_LLM_VERIFIER_TOP_P:-1.0}
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-8192}
export IF_LLM_VERIFIER_OMIT_MAX_TOKENS=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS:-true}
export IF_LLM_VERIFIER_REASONING_EFFORT=${IF_LLM_VERIFIER_REASONING_EFFORT:-}
export IF_LLM_VERIFIER_TIMEOUT=${IF_LLM_VERIFIER_TIMEOUT:-300}
export IF_LLM_VERIFIER_MAX_RETRIES=${IF_LLM_VERIFIER_MAX_RETRIES:-2}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-64}

# Keep per-worker native thread pools small; Ray already creates many workers.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

VERIFIER_PIDS=()
VERIFIER_LOG_FILES=()
cleanup_verifier() {
    for verifier_pid in "${VERIFIER_PIDS[@]:-}"; do
        kill -- "-${verifier_pid}" 2>/dev/null || kill "${verifier_pid}" 2>/dev/null || true
    done
    for verifier_pid in "${VERIFIER_PIDS[@]:-}"; do
        wait "${verifier_pid}" 2>/dev/null || true
    done
}
trap cleanup_verifier EXIT INT TERM

wait_for_verifier() {
    local verifier_base_url="${1:-${IF_LLM_VERIFIER_BASE_URL}}"
    local verifier_timeout="${2:-${IF_LLM_VERIFIER_WAIT_TIMEOUT}}"
    local verifier_pid="${3:-}"
    python3 - "${verifier_base_url}" "${verifier_timeout}" "${verifier_pid}" <<'PY'
import os
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
timeout = float(sys.argv[2])
pid = int(sys.argv[3]) if sys.argv[3] else None
if base_url.endswith("/v1"):
    url = f"{base_url}/models"
else:
    url = f"{base_url}/v1/models"

deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"IF LLM verifier process {pid} exited before becoming ready.", file=sys.stderr)
            sys.exit(2)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f"[IF LLM verifier] ready: {url}")
                sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    time.sleep(2)

print(f"Timed out waiting for IF LLM verifier at {url}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

if [[ "${IF_LLM_VERIFIER_REPLICA_SERVERS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    IFS=',' read -r -a IF_LLM_VERIFIER_GPU_ARRAY <<< "${IF_LLM_VERIFIER_GPU_SET}"
    IF_LLM_VERIFIER_PORT_ARRAY=()
    if [[ -n "${IF_LLM_VERIFIER_PORTS}" ]]; then
        IFS=',' read -r -a IF_LLM_VERIFIER_PORT_ARRAY <<< "${IF_LLM_VERIFIER_PORTS}"
    else
        for replica_idx in "${!IF_LLM_VERIFIER_GPU_ARRAY[@]}"; do
            IF_LLM_VERIFIER_PORT_ARRAY+=("$((IF_LLM_VERIFIER_PORT + replica_idx))")
        done
    fi
    if [[ "${#IF_LLM_VERIFIER_GPU_ARRAY[@]}" -ne "${#IF_LLM_VERIFIER_PORT_ARRAY[@]}" ]]; then
        echo "ERROR: IF_LLM_VERIFIER_PORTS count must match IF_LLM_VERIFIER_GPU_SET count." >&2
        exit 1
    fi

    IF_LLM_VERIFIER_BASE_URL_ARRAY=()
    if [[ -n "${IF_LLM_VERIFIER_BASE_URLS}" ]]; then
        IFS=',' read -r -a IF_LLM_VERIFIER_BASE_URL_ARRAY <<< "${IF_LLM_VERIFIER_BASE_URLS}"
    else
        for verifier_port in "${IF_LLM_VERIFIER_PORT_ARRAY[@]}"; do
            IF_LLM_VERIFIER_BASE_URL_ARRAY+=("http://${IF_LLM_VERIFIER_HOST}:${verifier_port}/v1")
        done
        IF_LLM_VERIFIER_BASE_URLS=$(IFS=,; echo "${IF_LLM_VERIFIER_BASE_URL_ARRAY[*]}")
        export IF_LLM_VERIFIER_BASE_URLS
    fi
    export IF_LLM_VERIFIER_BASE_URL="${IF_LLM_VERIFIER_BASE_URL_ARRAY[0]}"
else
    IF_LLM_VERIFIER_BASE_URLS="${IF_LLM_VERIFIER_BASE_URL}"
    export IF_LLM_VERIFIER_BASE_URLS
fi

if [[ "${IF_LLM_VERIFIER_START_SERVER}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ ! -x "${IF_LLM_VERIFIER_PYTHON}" ]]; then
        echo "ERROR: verifier Python is not executable: ${IF_LLM_VERIFIER_PYTHON}" >&2
        exit 1
    fi
    if ! "${IF_LLM_VERIFIER_PYTHON}" -c 'import vllm' >/dev/null; then
        echo "ERROR: vLLM is not importable with ${IF_LLM_VERIFIER_PYTHON}" >&2
        exit 1
    fi
    mkdir -p "${IF_LLM_VERIFIER_LOG_DIR}"
    if [[ "${IF_LLM_VERIFIER_REPLICA_SERVERS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        for replica_idx in "${!IF_LLM_VERIFIER_GPU_ARRAY[@]}"; do
            verifier_gpu=$(echo "${IF_LLM_VERIFIER_GPU_ARRAY[$replica_idx]}" | xargs)
            verifier_port=$(echo "${IF_LLM_VERIFIER_PORT_ARRAY[$replica_idx]}" | xargs)
            VERIFIER_LOG_FILE="${IF_LLM_VERIFIER_LOG_DIR}/gpt_oss_120b_gpu${verifier_gpu}_${verifier_port}.log"
            VERIFIER_CMD=(
                "${IF_LLM_VERIFIER_PYTHON}" -m vllm.entrypoints.cli.main serve "${IF_LLM_VERIFIER_MODEL}"
                --served-model-name "${IF_LLM_VERIFIER_MODEL}"
                --host "${IF_LLM_VERIFIER_HOST}"
                --port "${verifier_port}"
                --tensor-parallel-size "${IF_LLM_VERIFIER_TP}"
                --dtype "${IF_LLM_VERIFIER_DTYPE}"
                --gpu-memory-utilization "${IF_LLM_VERIFIER_GPU_MEM_UTIL}"
                --max-model-len "${IF_LLM_VERIFIER_MAX_MODEL_LEN}"
                --max-num-batched-tokens "${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS}"
                --max-num-seqs "${IF_LLM_VERIFIER_MAX_NUM_SEQS}"
            )
            if [[ "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
                VERIFIER_CMD+=(--enable-sleep-mode)
            fi
            if [[ "${IF_LLM_VERIFIER_TRUST_REMOTE_CODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
                VERIFIER_CMD+=(--trust-remote-code)
            fi

            setsid env CUDA_VISIBLE_DEVICES="${verifier_gpu}" VLLM_SERVER_DEV_MODE="${IF_LLM_VERIFIER_DEV_MODE}" \
                "${VERIFIER_CMD[@]}" >"${VERIFIER_LOG_FILE}" 2>&1 &
            verifier_pid=$!
            VERIFIER_PIDS+=("${verifier_pid}")
            VERIFIER_LOG_FILES+=("${VERIFIER_LOG_FILE}")
            echo "[IF LLM verifier] pid=${verifier_pid} gpu=${verifier_gpu} port=${verifier_port} log=${VERIFIER_LOG_FILE}" >&2
        done
    else
        VERIFIER_LOG_FILE="${IF_LLM_VERIFIER_LOG_FILE:-${IF_LLM_VERIFIER_LOG_DIR}/gpt_oss_120b_${IF_LLM_VERIFIER_PORT}.log}"
        VERIFIER_CMD=(
            "${IF_LLM_VERIFIER_PYTHON}" -m vllm.entrypoints.cli.main serve "${IF_LLM_VERIFIER_MODEL}"
            --served-model-name "${IF_LLM_VERIFIER_MODEL}"
            --host "${IF_LLM_VERIFIER_HOST}"
            --port "${IF_LLM_VERIFIER_PORT}"
            --tensor-parallel-size "${IF_LLM_VERIFIER_TP}"
            --data-parallel-size "${IF_LLM_VERIFIER_DP}"
            --dtype "${IF_LLM_VERIFIER_DTYPE}"
            --gpu-memory-utilization "${IF_LLM_VERIFIER_GPU_MEM_UTIL}"
            --max-model-len "${IF_LLM_VERIFIER_MAX_MODEL_LEN}"
            --max-num-batched-tokens "${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS}"
            --max-num-seqs "${IF_LLM_VERIFIER_MAX_NUM_SEQS}"
        )
        if [[ "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--enable-sleep-mode)
        fi
        if [[ "${IF_LLM_VERIFIER_TRUST_REMOTE_CODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--trust-remote-code)
        fi

        setsid env CUDA_VISIBLE_DEVICES="${IF_LLM_VERIFIER_GPU_SET}" VLLM_SERVER_DEV_MODE="${IF_LLM_VERIFIER_DEV_MODE}" \
            "${VERIFIER_CMD[@]}" >"${VERIFIER_LOG_FILE}" 2>&1 &
        verifier_pid=$!
        VERIFIER_PIDS+=("${verifier_pid}")
        VERIFIER_LOG_FILES+=("${VERIFIER_LOG_FILE}")
        echo "[IF LLM verifier] pid=${verifier_pid} gpu_set=${IF_LLM_VERIFIER_GPU_SET} log=${VERIFIER_LOG_FILE}" >&2
    fi
fi
IFS=',' read -r -a IF_LLM_VERIFIER_WAIT_URLS <<< "${IF_LLM_VERIFIER_BASE_URLS}"
for verifier_idx in "${!IF_LLM_VERIFIER_WAIT_URLS[@]}"; do
    verifier_base_url="${IF_LLM_VERIFIER_WAIT_URLS[$verifier_idx]}"
    verifier_pid="${VERIFIER_PIDS[$verifier_idx]:-}"
    if ! wait_for_verifier "${verifier_base_url}" "${IF_LLM_VERIFIER_WAIT_TIMEOUT}" "${verifier_pid}"; then
        verifier_log_file="${VERIFIER_LOG_FILES[$verifier_idx]:-}"
        if [[ -n "${verifier_log_file}" && -f "${verifier_log_file}" ]]; then
            echo "========== verifier log: ${verifier_log_file} ==========" >&2
            tail -100 "${verifier_log_file}" >&2
        fi
        exit 1
    fi
done
if [[ "${IF_LLM_VERIFIER_MANAGE_SLEEP}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ ! "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "ERROR: IF_LLM_VERIFIER_MANAGE_SLEEP=true requires IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=true." >&2
        exit 1
    fi
    for verifier_base_url in "${IF_LLM_VERIFIER_WAIT_URLS[@]}"; do
        IF_LLM_VERIFIER_CONTROL_URL="${verifier_base_url}" \
            bash "${SCRIPT_DIR}/control_qwen3_30ba3b_verifier.sh" sleep "${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    done
fi
########################### end verifier server ###########################

########################### actor training ###########################
export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

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
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-128}

# Disable PPL/anchor shaping. The only non-rule bonus is the LLM verifier bonus.
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.0}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-false}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}
export IF_REF_PPL_BASELINE=${IF_REF_PPL_BASELINE:-0}
export IF_REF_PPL_ANCHOR=${IF_REF_PPL_ANCHOR:-0}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_17b_grpo_nonthink_llmverifier_gptoss120b_bonus01_b1024_c1}

REWARD_MANAGER_PATH="${VERL_DIR}/if_rlvr/if_llm_verifier_reward_manager.py"

OVERRIDES=(
    reward.num_workers="${IF_LLM_VERIFIER_REWARD_WORKERS}"
    +reward.compute_after_rollout=true
    reward.reward_model.enable=False
    reward.reward_manager.source=importlib
    reward.reward_manager.name=IFLLMVerifierRewardManager
    reward.reward_manager.module.path="${REWARD_MANAGER_PATH}"
    custom_reward_function.path=null
    "+reward.reward_kwargs.verification_reward=1.0"
    "+reward.reward_kwargs.if_llm_verifier_base_url=${IF_LLM_VERIFIER_BASE_URL}"
    "+reward.reward_kwargs.if_llm_verifier_base_urls='${IF_LLM_VERIFIER_BASE_URLS}'"
    "+reward.reward_kwargs.if_llm_verifier_model=${IF_LLM_VERIFIER_MODEL}"
    "+reward.reward_kwargs.if_llm_verifier_bonus=${IF_LLM_VERIFIER_BONUS}"
    "+reward.reward_kwargs.if_llm_verifier_threshold=${IF_LLM_VERIFIER_THRESHOLD}"
    "+reward.reward_kwargs.if_llm_verifier_temperature=${IF_LLM_VERIFIER_TEMPERATURE}"
    "+reward.reward_kwargs.if_llm_verifier_top_p=${IF_LLM_VERIFIER_TOP_P}"
    "+reward.reward_kwargs.if_llm_verifier_max_tokens=${IF_LLM_VERIFIER_MAX_TOKENS}"
    "+reward.reward_kwargs.if_llm_verifier_omit_max_tokens=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS}"
    "+reward.reward_kwargs.if_llm_verifier_timeout=${IF_LLM_VERIFIER_TIMEOUT}"
    "+reward.reward_kwargs.if_llm_verifier_max_retries=${IF_LLM_VERIFIER_MAX_RETRIES}"
    "+reward.reward_kwargs.if_llm_verifier_response_format=${IF_LLM_VERIFIER_RESPONSE_FORMAT}"
    "+reward.reward_kwargs.if_llm_verifier_manage_sleep=${IF_LLM_VERIFIER_MANAGE_SLEEP}"
    "+reward.reward_kwargs.if_llm_verifier_sleep_level=${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    "+reward.reward_kwargs.if_llm_verifier_control_timeout=${IF_LLM_VERIFIER_CONTROL_TIMEOUT}"
)
if [[ -n "${IF_LLM_VERIFIER_REASONING_EFFORT}" ]]; then
    OVERRIDES+=("+reward.reward_kwargs.if_llm_verifier_reasoning_effort=${IF_LLM_VERIFIER_REASONING_EFFORT}")
fi

set +e
bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning_workspace.sh" "${OVERRIDES[@]}" "$@"
status=$?
set -e
exit "${status}"
