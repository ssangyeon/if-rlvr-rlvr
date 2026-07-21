#!/usr/bin/env bash
# Benchmark one 1024-prompt x 8-rollout IF-RLVR step on four B200 GPUs.

set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 CANONICAL_ROLLOUTS.jsonl OUTPUT_DIR" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INPUT_JSONL=$1
OUTPUT_ROOT=$2

PYTHON=${PYTHON:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/.miniforge3/envs/verl/bin/python}
VERL_DIR=${VERL_DIR:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr}
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_IDS=${GPU_IDS:-0,1,2,3}
PORT_BASE=${PORT_BASE:-28600}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1200}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
CONCURRENCY_PER_GPU=${CONCURRENCY_PER_GPU:-128}
MAX_RETRIES=${MAX_RETRIES:-0}

VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-40960}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
VLLM_PREFIX_CACHING=${VLLM_PREFIX_CACHING:-disabled}
REQUIRE_IDLE_GPUS=${REQUIRE_IDLE_GPUS:-1}
MAX_EXISTING_GPU_MEMORY_MIB=${MAX_EXISTING_GPU_MEMORY_MIB:-1024}

# Prompt-logprob scoring materializes vocabulary-wide logits. Keep its prefill
# batches smaller than verifier batches and leave headroom outside the KV cache.
PPL_CONCURRENCY_PER_GPU=${PPL_CONCURRENCY_PER_GPU:-16}
PPL_VLLM_GPU_MEMORY_UTILIZATION=${PPL_VLLM_GPU_MEMORY_UTILIZATION:-0.75}
PPL_VLLM_MAX_NUM_BATCHED_TOKENS=${PPL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}
PPL_VLLM_MAX_NUM_SEQS=${PPL_VLLM_MAX_NUM_SEQS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PPL_MODEL=${PPL_MODEL:-Qwen/Qwen3-4B}
GPT_OSS_MODEL=${GPT_OSS_MODEL:-openai/gpt-oss-120b}
QWEN30_MODEL=${QWEN30_MODEL:-Qwen/Qwen3-30B-A3B}
QWEN4_VERIFIER_MODEL=${QWEN4_VERIFIER_MODEL:-Qwen/Qwen3-4B}

JUDGE_MODE=${JUDGE_MODE:-all}
RUN_PPL=${RUN_PPL:-1}
RUN_GPT_OSS=${RUN_GPT_OSS:-1}
RUN_QWEN30=${RUN_QWEN30:-1}
RUN_QWEN4_VERIFIER=${RUN_QWEN4_VERIFIER:-1}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != 4 )); then
    echo "ERROR: GPU_IDS must contain exactly four GPUs; got: ${GPU_IDS}" >&2
    exit 2
fi

FIRST_GPU=$(echo "${GPU_ARRAY[0]}" | xargs)
FIRST_GPU_NAME=$(nvidia-smi --id="${FIRST_GPU}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs || true)
HARDWARE_LABEL=${HARDWARE_LABEL:-"4x ${FIRST_GPU_NAME:-GPU}"}
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/server_logs"

MODEL_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    MODEL_ARGS+=(--local-files-only)
fi

SERVER_CACHE_ARGS=()
case "${VLLM_PREFIX_CACHING}" in
    disabled) SERVER_CACHE_ARGS+=(--no-enable-prefix-caching) ;;
    enabled) SERVER_CACHE_ARGS+=(--enable-prefix-caching) ;;
    *)
        echo "ERROR: VLLM_PREFIX_CACHING must be 'disabled' or 'enabled'." >&2
        exit 2
        ;;
esac

COMMON_CLIENT_ARGS=(
    --input "${INPUT_JSONL}"
    --expected-responses 8192
    --expected-prompts 1024
    --rollouts-per-prompt 8
    --timeout "${REQUEST_TIMEOUT}"
    --max-retries "${MAX_RETRIES}"
    --hardware "${HARDWARE_LABEL}"
    --prefix-caching "${VLLM_PREFIX_CACHING}"
    "${MODEL_ARGS[@]}"
)

CURRENT_PIDS=()
CURRENT_ENDPOINTS=()

cleanup_replicas() {
    local pid
    for pid in "${CURRENT_PIDS[@]:-}"; do
        kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    done
    for pid in "${CURRENT_PIDS[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
    CURRENT_PIDS=()
    CURRENT_ENDPOINTS=()
}
trap cleanup_replicas EXIT INT TERM

wait_for_replica() {
    local base_url=$1
    local pid=$2
    local deadline=$((SECONDS + WAIT_TIMEOUT))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: vLLM pid=${pid} exited before becoming healthy (${base_url})." >&2
            return 1
        fi
        if curl --silent --show-error --fail --max-time 5 "${base_url}/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "ERROR: timed out waiting for ${base_url}" >&2
    return 1
}

require_idle_gpus() {
    [[ "${REQUIRE_IDLE_GPUS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]] || return 0
    local gpu used_mib
    for gpu in "${GPU_ARRAY[@]}"; do
        gpu=$(echo "${gpu}" | xargs)
        used_mib=$(nvidia-smi \
            --id="${gpu}" \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits | head -1 | xargs)
        if (( used_mib > MAX_EXISTING_GPU_MEMORY_MIB )); then
            echo "ERROR: GPU ${gpu} already uses ${used_mib} MiB; refusing to overlap benchmark replicas." >&2
            return 1
        fi
    done
}

start_replicas() {
    local slug=$1
    local model=$2
    local first_port=$3
    local gpu_memory_utilization=${4:-${VLLM_GPU_MEMORY_UTILIZATION}}
    local max_num_batched_tokens=${5:-${VLLM_MAX_NUM_BATCHED_TOKENS}}
    local max_num_seqs=${6:-${VLLM_MAX_NUM_SEQS}}
    local replica_index gpu port log_file pid endpoint
    local started_at=${SECONDS}

    cleanup_replicas
    require_idle_gpus
    echo "[server] starting ${slug}: model=${model}, TP=1, replicas=4, GPUs=${GPU_IDS}, gpu_memory_utilization=${gpu_memory_utilization}, max_num_batched_tokens=${max_num_batched_tokens}, max_num_seqs=${max_num_seqs}"
    for replica_index in "${!GPU_ARRAY[@]}"; do
        gpu=$(echo "${GPU_ARRAY[$replica_index]}" | xargs)
        port=$((first_port + replica_index))
        log_file="${OUTPUT_ROOT}/server_logs/${slug}_gpu${gpu}_port${port}.log"
        setsid env CUDA_VISIBLE_DEVICES="${gpu}" \
            "${PYTHON}" -m vllm.entrypoints.cli.main serve "${model}" \
                --served-model-name "${model}" \
                --host 127.0.0.1 \
                --port "${port}" \
                --tensor-parallel-size 1 \
                --dtype "${VLLM_DTYPE}" \
                --gpu-memory-utilization "${gpu_memory_utilization}" \
                --max-model-len "${VLLM_MAX_MODEL_LEN}" \
                --max-num-batched-tokens "${max_num_batched_tokens}" \
                --max-num-seqs "${max_num_seqs}" \
                --enable-prompt-tokens-details \
                "${SERVER_CACHE_ARGS[@]}" \
            >"${log_file}" 2>&1 &
        pid=$!
        CURRENT_PIDS+=("${pid}")
        endpoint="http://127.0.0.1:${port}/v1"
        CURRENT_ENDPOINTS+=("${endpoint}")
        echo "[server] gpu=${gpu} pid=${pid} endpoint=${endpoint} log=${log_file}"
    done

    for replica_index in "${!CURRENT_ENDPOINTS[@]}"; do
        if ! wait_for_replica "${CURRENT_ENDPOINTS[$replica_index]}" "${CURRENT_PIDS[$replica_index]}"; then
            tail -n 100 -- "${OUTPUT_ROOT}/server_logs/${slug}_"*.log >&2 || true
            return 1
        fi
    done
    echo "[server] all ${slug} replicas healthy after $((SECONDS - started_at))s; load time is excluded."
}

join_endpoints() {
    local IFS=,
    echo "${CURRENT_ENDPOINTS[*]}"
}

"${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" validate \
    --input "${INPUT_JSONL}" \
    --expected-responses 8192 \
    --expected-prompts 1024 \
    --rollouts-per-prompt 8

REPORT_INPUTS=()

run_is_complete() {
    local output_dir=$1
    local expected_method=$2
    local expected_model=$3
    local expected_concurrency=$4
    local expected_gpu_memory_utilization=$5
    local expected_max_model_len=$6
    local expected_max_num_batched_tokens=$7
    local expected_max_num_seqs=$8
    local metrics_path="${output_dir}/metrics.json"
    local results_path="${output_dir}/results.jsonl"

    [[ -s "${metrics_path}" && -s "${results_path}" ]] || return 1
    "${PYTHON}" - \
        "${metrics_path}" \
        "${expected_method}" \
        "${expected_model}" \
        "${expected_concurrency}" \
        "${expected_gpu_memory_utilization}" \
        "${expected_max_model_len}" \
        "${expected_max_num_batched_tokens}" \
        "${expected_max_num_seqs}" <<'PY'
import json
import sys

(
    metrics_path,
    expected_method,
    expected_model,
    expected_concurrency,
    expected_gpu_memory_utilization,
    expected_max_model_len,
    expected_max_num_batched_tokens,
    expected_max_num_seqs,
) = sys.argv[1:]
try:
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    server_config = metrics.get("server_config") or {}
    complete = (
        metrics.get("method") == expected_method
        and metrics.get("model") == expected_model
        and int(metrics.get("client_concurrency_per_endpoint", -1)) == int(expected_concurrency)
        and float(server_config.get("gpu_memory_utilization", -1.0))
        == float(expected_gpu_memory_utilization)
        and int(server_config.get("max_model_len", -1)) == int(expected_max_model_len)
        and int(server_config.get("max_num_batched_tokens", -1))
        == int(expected_max_num_batched_tokens)
        and int(server_config.get("max_num_seqs", -1)) == int(expected_max_num_seqs)
        and int(metrics.get("input_response_count", -1)) == 8192
        and int(metrics.get("error_count", -1)) == 0
        and int(metrics.get("success_count", -1))
        == int(metrics.get("selected_response_count", -2))
    )
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    complete = False
raise SystemExit(0 if complete else 1)
PY
}

if [[ "${RUN_PPL}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if run_is_complete \
        "${OUTPUT_ROOT}/qwen3_4b_ppl" \
        qwen3_4b_ppl \
        "${PPL_MODEL}" \
        "${PPL_CONCURRENCY_PER_GPU}" \
        "${PPL_VLLM_GPU_MEMORY_UTILIZATION}" \
        "${VLLM_MAX_MODEL_LEN}" \
        "${PPL_VLLM_MAX_NUM_BATCHED_TOKENS}" \
        "${PPL_VLLM_MAX_NUM_SEQS}"; then
        echo "[resume] skipping completed qwen3_4b_ppl"
    else
        start_replicas \
            qwen3_4b_ppl \
            "${PPL_MODEL}" \
            "${PORT_BASE}" \
            "${PPL_VLLM_GPU_MEMORY_UTILIZATION}" \
            "${PPL_VLLM_MAX_NUM_BATCHED_TOKENS}" \
            "${PPL_VLLM_MAX_NUM_SEQS}"
        ENDPOINTS=$(join_endpoints)
        "${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" ppl \
            "${COMMON_CLIENT_ARGS[@]}" \
            --concurrency-per-endpoint "${PPL_CONCURRENCY_PER_GPU}" \
            --server-gpu-memory-utilization "${PPL_VLLM_GPU_MEMORY_UTILIZATION}" \
            --server-max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --server-max-num-batched-tokens "${PPL_VLLM_MAX_NUM_BATCHED_TOKENS}" \
            --server-max-num-seqs "${PPL_VLLM_MAX_NUM_SEQS}" \
            --output "${OUTPUT_ROOT}/qwen3_4b_ppl" \
            --method qwen3_4b_ppl \
            --model "${PPL_MODEL}" \
            --endpoints "${ENDPOINTS}"
        cleanup_replicas
    fi
    REPORT_INPUTS+=("${OUTPUT_ROOT}/qwen3_4b_ppl")
fi

if [[ "${RUN_QWEN4_VERIFIER}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if run_is_complete \
        "${OUTPUT_ROOT}/qwen3_4b_verifier" \
        qwen3_4b_verifier \
        "${QWEN4_VERIFIER_MODEL}" \
        "${CONCURRENCY_PER_GPU}" \
        "${VLLM_GPU_MEMORY_UTILIZATION}" \
        "${VLLM_MAX_MODEL_LEN}" \
        "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
        "${VLLM_MAX_NUM_SEQS}"; then
        echo "[resume] skipping completed qwen3_4b_verifier"
    else
        # Use fresh replicas so the PPL run cannot pre-warm verifier kernels/state.
        start_replicas qwen3_4b_verifier "${QWEN4_VERIFIER_MODEL}" "$((PORT_BASE + 5))"
        ENDPOINTS=$(join_endpoints)
        "${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" verifier \
            "${COMMON_CLIENT_ARGS[@]}" \
            --concurrency-per-endpoint "${CONCURRENCY_PER_GPU}" \
            --server-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
            --server-max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --server-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
            --server-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
            --output "${OUTPUT_ROOT}/qwen3_4b_verifier" \
            --method qwen3_4b_verifier \
            --model "${QWEN4_VERIFIER_MODEL}" \
            --endpoints "${ENDPOINTS}" \
            --judge-mode "${JUDGE_MODE}" \
            --enable-thinking false \
            --max-tokens 2048
        cleanup_replicas
    fi
    REPORT_INPUTS+=("${OUTPUT_ROOT}/qwen3_4b_verifier")
fi

if [[ "${RUN_QWEN30}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if run_is_complete \
        "${OUTPUT_ROOT}/qwen3_30b_a3b_verifier" \
        qwen3_30b_a3b_verifier \
        "${QWEN30_MODEL}" \
        "${CONCURRENCY_PER_GPU}" \
        "${VLLM_GPU_MEMORY_UTILIZATION}" \
        "${VLLM_MAX_MODEL_LEN}" \
        "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
        "${VLLM_MAX_NUM_SEQS}"; then
        echo "[resume] skipping completed qwen3_30b_a3b_verifier"
    else
        start_replicas qwen3_30b_a3b "${QWEN30_MODEL}" "$((PORT_BASE + 10))"
        ENDPOINTS=$(join_endpoints)
        "${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" verifier \
            "${COMMON_CLIENT_ARGS[@]}" \
            --concurrency-per-endpoint "${CONCURRENCY_PER_GPU}" \
            --server-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
            --server-max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --server-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
            --server-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
            --output "${OUTPUT_ROOT}/qwen3_30b_a3b_verifier" \
            --method qwen3_30b_a3b_verifier \
            --model "${QWEN30_MODEL}" \
            --endpoints "${ENDPOINTS}" \
            --judge-mode "${JUDGE_MODE}" \
            --enable-thinking false \
            --max-tokens 2048
        cleanup_replicas
    fi
    REPORT_INPUTS+=("${OUTPUT_ROOT}/qwen3_30b_a3b_verifier")
fi

if [[ "${RUN_GPT_OSS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if run_is_complete \
        "${OUTPUT_ROOT}/gpt_oss_120b_verifier" \
        gpt_oss_120b_verifier \
        "${GPT_OSS_MODEL}" \
        "${CONCURRENCY_PER_GPU}" \
        "${VLLM_GPU_MEMORY_UTILIZATION}" \
        "${VLLM_MAX_MODEL_LEN}" \
        "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
        "${VLLM_MAX_NUM_SEQS}"; then
        echo "[resume] skipping completed gpt_oss_120b_verifier"
    else
        start_replicas gpt_oss_120b "${GPT_OSS_MODEL}" "$((PORT_BASE + 20))"
        ENDPOINTS=$(join_endpoints)
        "${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" verifier \
            "${COMMON_CLIENT_ARGS[@]}" \
            --concurrency-per-endpoint "${CONCURRENCY_PER_GPU}" \
            --server-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
            --server-max-model-len "${VLLM_MAX_MODEL_LEN}" \
            --server-max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
            --server-max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
            --output "${OUTPUT_ROOT}/gpt_oss_120b_verifier" \
            --method gpt_oss_120b_verifier \
            --model "${GPT_OSS_MODEL}" \
            --endpoints "${ENDPOINTS}" \
            --judge-mode "${JUDGE_MODE}" \
            --enable-thinking none \
            --omit-max-tokens
        cleanup_replicas
    fi
    REPORT_INPUTS+=("${OUTPUT_ROOT}/gpt_oss_120b_verifier")
fi

if (( ${#REPORT_INPUTS[@]} == 0 )); then
    echo "ERROR: no benchmark was enabled." >&2
    exit 2
fi

"${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" report \
    --metrics "${REPORT_INPUTS[@]}" \
    --output "${OUTPUT_ROOT}/report"

echo "[done] ${OUTPUT_ROOT}/report/report.md"
