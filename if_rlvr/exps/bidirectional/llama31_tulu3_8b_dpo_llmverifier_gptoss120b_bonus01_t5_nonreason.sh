#!/usr/bin/env bash
# GRPO | allenai/Llama-3.1-Tulu-3-8B-DPO (non-reasoning)
#   + openai/gpt-oss-120b LLM-verifier bonus at score threshold 5.
#   NO anchor / PPL reward shaping - the judge is the only non-rule signal.
#
# Tulu 3 counterpart of qwen3_17b_llmverifier_gptoss120b_bonus01_nonreason.sh,
# sized for one node of 8x H100 80GB. Structurally identical to
# llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh - only
# the judge model, its serving knobs, and the score threshold differ.
#
# ---------------------------------------------------------------------------
# Per-rollout reward
# ---------------------------------------------------------------------------
#   1. policy rollout on GPUs 0-7 (vLLM, TP=1 per GPU)
#   2. IFEval constraint reward; rows with constraint=0 stop here
#   3. every constraint-positive rollout gets ONE gpt-oss-120b judge call;
#      score >= 5 adds +0.1 on top of the constraint reward
#
# ---------------------------------------------------------------------------
# GPU plan - one model per GPU, no tensor sharding anywhere
# ---------------------------------------------------------------------------
#   * 8 judge replicas, TP=1, one gpt-oss-120b per GPU. Its MXFP4 weights are
#     65.2 GB (confirmed via HfApi file listing, excluding the bf16 `original/`
#     duplicate that ignore_patterns already skips at download time) - this
#     fits a single 80 GB H100 with no tensor-parallel sharding, exactly like
#     the existing qwen3_17b_llmverifier_gptoss120b_bonus01_nonreason.sh and
#     qwen3_4b_t4b_anchor_pyx01_llmverifier_gptoss120b_bonus01_reasoning.sh
#     precedents in this same repo. TP=2 (sharding across 2 GPUs) would only
#     halve the number of concurrent replicas for no memory benefit.
#   * They start on empty GPUs and are put to sleep (level 1: weights to host
#     RAM, KV freed) before the trainer touches the GPUs, exactly as in the
#     Qwen3-30B-A3B sibling script.
#   * The reward loop wakes them only for the judge phase and sleeps them again
#     in a `finally`, so judge weights and the training footprint are never
#     resident together.
#
# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
#   * wandb: trainer.logger=[console,wandb], ${WANDB_ENTITY}/${PROJECT_NAME}.
#   * Hugging Face Hub upload is NOT handled by this script (unlike the
#     Qwen3-30B-A3B sibling) - point push_checkpoints_to_hf.py at ${CKPT_DIR}
#     yourself, or use the orchestrator that calls this script.
#
# Usage:  bash llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh
# Any extra arguments are forwarded verbatim as Hydra overrides.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
DATA_ROOT=$(dirname -- "${VERL_DIR}")
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
cd "${VERL_DIR}"

########################### model / cache placement ###########################
# Tulu 3 8B (16 GB) plus gpt-oss-120b (65 GB) need ~85 GB of HF cache.
IF_HF_CACHE_MIN_GB=${IF_HF_CACHE_MIN_GB:-120}
HF_HOME=${HF_HOME:-${DATA_ROOT}/.cache/huggingface}
free_gb() {
    local dir=$1
    while [[ ! -d "${dir}" && "${dir}" != "/" ]]; do dir=$(dirname -- "${dir}"); done
    df -Pk "${dir}" 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}' | grep -E '^[0-9]+$' || echo 0
}
if [[ "$(free_gb "${HF_HOME}")" -lt "${IF_HF_CACHE_MIN_GB}" ]]; then
    HF_HOME_FALLBACK="${DATA_ROOT}/.cache/huggingface"
    if [[ "${HF_HOME%/}" != "${HF_HOME_FALLBACK%/}" && "$(free_gb "${HF_HOME_FALLBACK}")" -ge "${IF_HF_CACHE_MIN_GB}" ]]; then
        echo "[setup] HF_HOME=${HF_HOME} has $(free_gb "${HF_HOME}") GB free (< ${IF_HF_CACHE_MIN_GB} GB);" >&2
        echo "[setup] switching to ${HF_HOME_FALLBACK}. Set IF_HF_CACHE_MIN_GB=0 to keep the original." >&2
        HF_HOME="${HF_HOME_FALLBACK}"
    else
        echo "ERROR: no HF cache location with >= ${IF_HF_CACHE_MIN_GB} GB free (checked ${HF_HOME})." >&2
        echo "       Point HF_HOME at a disk that can hold Tulu-3-8B (16 GB) + gpt-oss-120b (65 GB)." >&2
        exit 1
    fi
fi
export HF_HOME
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${CACHE_ROOT}"

########################### run isolation ###########################
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_8b_dpo_gptoss120b_t5_slot${RUN_SLOT}}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-$((20000 + RUN_SLOT * 2000))}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-$((IF_RLVR_PORT_BASE + 200))}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

########################### Tulu 3 runtime + defaults ###########################
export GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.72}

if [[ -z "${TULU3_PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    if "${CONDA_PREFIX}/bin/python" -c 'import vllm' >/dev/null 2>&1; then
        export TULU3_CONDA_ENV_DIR="${CONDA_PREFIX}"
        export TULU3_CONDA_SH="${TULU3_CONDA_SH:-$(dirname -- "$(dirname -- "${CONDA_PREFIX}")")/etc/profile.d/conda.sh}"
        export TULU3_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    fi
fi

# shellcheck source=_llama31_tulu3_8b_common.sh
source "${SCRIPT_DIR}/_llama31_tulu3_8b_common.sh"
tulu3_require_runtime

"${TULU3_PYTHON_BIN}" - <<'PY'
import sys

try:
    from vllm.v1.worker.gpu_worker import Worker  # noqa: F401
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"[preflight] vLLM engine worker stack is not importable: {type(exc).__name__}: {exc}\n"
        "            Every vLLM engine in this run would fail the same way. Fix the "
        "environment first\n"
        "            (e.g. vLLM 0.11 pins numba==0.61.2, which requires numpy<2.3)."
    )
print("[preflight] vLLM engine worker stack imports cleanly")
PY

########################### training hyper-parameters ###########################
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export SAVE_FREQ=${SAVE_FREQ:-91}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-32}

# Host-side parameter/optimizer residency, so the awake judge fits on the same
# GPU as the (idled) trainer - see llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh
# for the full measurement writeup; identical reasoning applies here.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}

########################### reward: constraint + LLM verifier only ###########################
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.0}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-false}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_PPL_BASELINE=${IF_REF_PPL_BASELINE:-0}
export IF_REF_PPL_ANCHOR=${IF_REF_PPL_ANCHOR:-0}

########################### LLM verifier ###########################
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-openai/gpt-oss-120b}
export IF_LLM_VERIFIER_REVISION=${IF_LLM_VERIFIER_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-${GPU_SET}}
export IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP:-1}
export IF_LLM_VERIFIER_HOST=${IF_LLM_VERIFIER_HOST:-127.0.0.1}
export IF_LLM_VERIFIER_PORT=${IF_LLM_VERIFIER_PORT:-$((IF_RLVR_PORT_BASE + 1200))}
export IF_LLM_VERIFIER_START_SERVER=${IF_LLM_VERIFIER_START_SERVER:-true}
export IF_LLM_VERIFIER_PYTHON=${IF_LLM_VERIFIER_PYTHON:-${TULU3_PYTHON_BIN}}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier}

export IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE:-true}
export IF_LLM_VERIFIER_MANAGE_SLEEP=${IF_LLM_VERIFIER_MANAGE_SLEEP:-true}
export IF_LLM_VERIFIER_SLEEP_LEVEL=${IF_LLM_VERIFIER_SLEEP_LEVEL:-1}
export IF_LLM_VERIFIER_DEV_MODE=${IF_LLM_VERIFIER_DEV_MODE:-1}
export IF_LLM_VERIFIER_CONTROL_TIMEOUT=${IF_LLM_VERIFIER_CONTROL_TIMEOUT:-600}
export IF_LLM_VERIFIER_WAIT_TIMEOUT=${IF_LLM_VERIFIER_WAIT_TIMEOUT:-2400}

# gpt-oss-120b's MXFP4 weights are 65.2 GB; 0.85 gives ~69.3 GB of vLLM-managed
# budget, matching the utilisation proven for the similar-sized (61.1 GB)
# Qwen3-30B-A3B judge in the sibling script - but that vLLM-managed budget is
# NOT the whole story for this model. Its Triton `matmul_ogs` MoE kernel
# (vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py) does a
# raw torch.empty() for its output workspace OUTSIDE vLLM's memory pool, sized
# by however many tokens are in that forward call; this repo's first attempt
# at 32768/128 measured that workspace pushing actual usage to ~72-73 GiB -
# already past the nominal 0.85 budget - and a live run OOM'd
# (torch.OutOfMemoryError, "Tried to allocate 900.00 MiB" / "1.96 GiB") the
# moment the trainer's own (small but nonzero, ~2-6 GiB observed) resident
# footprint landed on the same GPU at the same time. GPU_MEM_UTIL can't go
# much lower without leaving no room for the 65.2 GB of weights themselves, so
# the fix is on the other side: cut max_num_batched_tokens/max_num_seqs so the
# MoE forward pass this workspace scales with is smaller, giving real headroom
# instead of running flush against the card. No --enforce-eager: the existing
# gpt-oss-120b precedents in this repo (qwen3_17b_llmverifier_gptoss120b_bonus01_nonreason.sh,
# qwen3_4b_t4b_anchor_pyx01_llmverifier_gptoss120b_bonus01_reasoning.sh) both
# leave CUDA graphs on for this exact model, and graph capture itself wasn't
# implicated in the OOM.
export IF_LLM_VERIFIER_DTYPE=${IF_LLM_VERIFIER_DTYPE:-bfloat16}
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.85}
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-24576}
export IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS=${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS:-8192}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-64}
export IF_LLM_VERIFIER_ENFORCE_EAGER=${IF_LLM_VERIFIER_ENFORCE_EAGER:-false}
export IF_LLM_VERIFIER_TRUST_REMOTE_CODE=${IF_LLM_VERIFIER_TRUST_REMOTE_CODE:-false}
export IF_LLM_VERIFIER_USE_SYMM_MEM=${IF_LLM_VERIFIER_USE_SYMM_MEM:-0}

# Reward-side judging policy. gpt-oss-120b answers through its harmony format
# (reasoning/analysis channel + final channel); the reward manager extracts
# the final channel itself, so no chat_template_kwargs (enable_thinking) is
# needed here - that is a Qwen3-hybrid-model-specific knob. Longer max_tokens
# and no truncation cap accommodate its reasoning tokens before the final
# {"Score": N}, matching qwen3_17b_llmverifier_gptoss120b_bonus01_nonreason.sh.
export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-5}
export IF_LLM_VERIFIER_TEMPERATURE=${IF_LLM_VERIFIER_TEMPERATURE:-0.0}
export IF_LLM_VERIFIER_TOP_P=${IF_LLM_VERIFIER_TOP_P:-1.0}
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-8192}
export IF_LLM_VERIFIER_OMIT_MAX_TOKENS=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS:-true}
export IF_LLM_VERIFIER_REASONING_EFFORT=${IF_LLM_VERIFIER_REASONING_EFFORT:-}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}
export IF_LLM_VERIFIER_TIMEOUT=${IF_LLM_VERIFIER_TIMEOUT:-300}
export IF_LLM_VERIFIER_MAX_RETRIES=${IF_LLM_VERIFIER_MAX_RETRIES:-2}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-64}

########################### checkpoints, wandb ###########################
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
IF_LLM_VERIFIER_BONUS_TAG=${IF_LLM_VERIFIER_BONUS/./}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_llmverifier_gptoss120b_nonthink_bonus${IF_LLM_VERIFIER_BONUS_TAG}_threshold${IF_LLM_VERIFIER_THRESHOLD}_b${TRAIN_BATCH_SIZE}_c1_t1_2k}

CKPT_DIR=${CKPT_DIR:-${VERL_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${CKPT_DIR}"

if [[ -z "${WANDB_API_KEY:-}" && ! -s "${HOME}/.netrc" ]]; then
    echo "ERROR: wandb logging is required but no WANDB_API_KEY and no ~/.netrc were found." >&2
    exit 1
fi

########################### banner ###########################
cat >&2 <<BANNER
[run] experiment      : ${EXPERIMENT_NAME}
[run] policy          : ${MODEL_PATH} @ ${TULU3_MODEL_REVISION}
[run] verifier        : ${IF_LLM_VERIFIER_MODEL} (threshold=${IF_LLM_VERIFIER_THRESHOLD}, bonus=${IF_LLM_VERIFIER_BONUS})
[run] reward          : IFEval constraint + judge bonus (no anchor/PPL shaping)
[run] gpus            : train=${GPU_SET} verifier=${IF_LLM_VERIFIER_GPU_SET} (tp=${IF_LLM_VERIFIER_TP})
[run] batch           : ${TRAIN_BATCH_SIZE} x n=${ROLLOUT_N}, ${TOTAL_EPOCHS} epochs, save every ${SAVE_FREQ} steps
[run] checkpoints     : ${CKPT_DIR}
[run] wandb           : ${WANDB_ENTITY}/${PROJECT_NAME}
BANNER

########################### weight prefetch ###########################
echo "[setup] prefetching model weights into ${HF_HUB_CACHE}"
"${TULU3_PYTHON_BIN}" - <<'PY'
import os

from huggingface_hub import snapshot_download

IGNORE = ["*.pth", "*.bin", "*.bin.index.json", "original/*", "consolidated*"]

targets = [(os.environ["MODEL_PATH"], None)]
verifier = os.environ["IF_LLM_VERIFIER_MODEL"]
if not os.path.isdir(verifier):
    targets.append((verifier, os.environ.get("IF_LLM_VERIFIER_REVISION") or None))

for repo_id, revision in targets:
    if os.path.isdir(repo_id):
        continue
    path = snapshot_download(repo_id, revision=revision, ignore_patterns=IGNORE)
    print(f"[setup] ready: {repo_id} -> {path}")
PY

# gpt-oss-120b's OpenAI-compatible "responses" endpoint loads its harmony
# tiktoken encoding from a content-addressed cache file under
# /tmp/tiktoken-rs-cache/<hash> on first use. That cache write is not atomic
# against concurrent writers, and all 8 replicas below hit the same encoding
# (same hash) at once - one process's partial/interleaved write corrupts the
# file for everyone else, surfacing minutes later as
# "openai_harmony.HarmonyError: invalid tiktoken vocab file" during API
# server startup (after the multi-minute engine init already completed).
# Populate it once, sequentially, before any replica starts.
echo "[setup] pre-warming openai_harmony encoding cache (avoids an 8-way concurrent-write race)"
"${TULU3_PYTHON_BIN}" -c "
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print('[setup] harmony encoding cache ready')
"

########################### verifier servers ###########################
VERIFIER_PIDS=()
VERIFIER_LOG_FILES=()
VERIFIER_BASE_URL_ARRAY=()

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ((${#VERIFIER_PIDS[@]})); then
        for pid in "${VERIFIER_PIDS[@]}"; do
            kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        done
        for pid in "${VERIFIER_PIDS[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
        VERIFIER_PIDS=()
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

wait_for_verifier() {
    local base_url=$1
    local timeout=$2
    local pid=${3:-}
    "${TULU3_PYTHON_BIN}" - "${base_url}" "${timeout}" "${pid}" <<'PY'
import os
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
timeout = float(sys.argv[2])
pid = int(sys.argv[3]) if sys.argv[3] else None
url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"

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

IFS=',' read -r -a VERIFIER_GPU_ARRAY <<< "${IF_LLM_VERIFIER_GPU_SET//[[:space:]]/}"
if ((${#VERIFIER_GPU_ARRAY[@]} % IF_LLM_VERIFIER_TP != 0)); then
    echo "ERROR: IF_LLM_VERIFIER_GPU_SET (${#VERIFIER_GPU_ARRAY[@]} GPUs) is not divisible by IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP}." >&2
    exit 1
fi
VERIFIER_REPLICAS=$((${#VERIFIER_GPU_ARRAY[@]} / IF_LLM_VERIFIER_TP))

for ((replica = 0; replica < VERIFIER_REPLICAS; replica++)); do
    VERIFIER_BASE_URL_ARRAY+=("http://${IF_LLM_VERIFIER_HOST}:$((IF_LLM_VERIFIER_PORT + replica))/v1")
done
IF_LLM_VERIFIER_BASE_URLS=$(IFS=,; echo "${VERIFIER_BASE_URL_ARRAY[*]}")
export IF_LLM_VERIFIER_BASE_URLS
export IF_LLM_VERIFIER_BASE_URL="${VERIFIER_BASE_URL_ARRAY[0]}"

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

    for ((replica = 0; replica < VERIFIER_REPLICAS; replica++)); do
        replica_gpus=()
        for ((lane = 0; lane < IF_LLM_VERIFIER_TP; lane++)); do
            replica_gpus+=("${VERIFIER_GPU_ARRAY[$((replica * IF_LLM_VERIFIER_TP + lane))]}")
        done
        replica_gpu_csv=$(IFS=,; echo "${replica_gpus[*]}")
        replica_port=$((IF_LLM_VERIFIER_PORT + replica))
        replica_log="${IF_LLM_VERIFIER_LOG_DIR}/gpt_oss_120b_gpu${replica_gpu_csv//,/_}_${replica_port}.log"

        VERIFIER_CMD=(
            "${IF_LLM_VERIFIER_PYTHON}" -m vllm.entrypoints.cli.main serve "${IF_LLM_VERIFIER_MODEL}"
            --served-model-name "${IF_LLM_VERIFIER_MODEL}"
            --host "${IF_LLM_VERIFIER_HOST}"
            --port "${replica_port}"
            --tensor-parallel-size "${IF_LLM_VERIFIER_TP}"
            --dtype "${IF_LLM_VERIFIER_DTYPE}"
            --gpu-memory-utilization "${IF_LLM_VERIFIER_GPU_MEM_UTIL}"
            --max-model-len "${IF_LLM_VERIFIER_MAX_MODEL_LEN}"
            --max-num-batched-tokens "${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS}"
            --max-num-seqs "${IF_LLM_VERIFIER_MAX_NUM_SEQS}"
        )
        if [[ -n "${IF_LLM_VERIFIER_REVISION}" && ! -d "${IF_LLM_VERIFIER_MODEL}" ]]; then
            VERIFIER_CMD+=(--revision "${IF_LLM_VERIFIER_REVISION}" --tokenizer-revision "${IF_LLM_VERIFIER_REVISION}")
        fi
        if [[ "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--enable-sleep-mode)
        fi
        if [[ "${IF_LLM_VERIFIER_ENFORCE_EAGER}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--enforce-eager)
        fi
        if [[ "${IF_LLM_VERIFIER_TRUST_REMOTE_CODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--trust-remote-code)
        fi

        setsid env CUDA_VISIBLE_DEVICES="${replica_gpu_csv}" \
            VLLM_SERVER_DEV_MODE="${IF_LLM_VERIFIER_DEV_MODE}" \
            VLLM_ALLREDUCE_USE_SYMM_MEM="${IF_LLM_VERIFIER_USE_SYMM_MEM}" \
            "${VERIFIER_CMD[@]}" >"${replica_log}" 2>&1 &
        verifier_pid=$!
        VERIFIER_PIDS+=("${verifier_pid}")
        VERIFIER_LOG_FILES+=("${replica_log}")
        echo "[IF LLM verifier] pid=${verifier_pid} gpu=${replica_gpu_csv} port=${replica_port} log=${replica_log}" >&2
    done
fi

for replica in "${!VERIFIER_BASE_URL_ARRAY[@]}"; do
    if ! wait_for_verifier "${VERIFIER_BASE_URL_ARRAY[$replica]}" "${IF_LLM_VERIFIER_WAIT_TIMEOUT}" "${VERIFIER_PIDS[$replica]:-}"; then
        replica_log="${VERIFIER_LOG_FILES[$replica]:-}"
        if [[ -n "${replica_log}" && -f "${replica_log}" ]]; then
            echo "========== verifier log: ${replica_log} ==========" >&2
            tail -100 "${replica_log}" >&2
        fi
        exit 1
    fi
done

if [[ "${IF_LLM_VERIFIER_MANAGE_SLEEP}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ ! "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "ERROR: IF_LLM_VERIFIER_MANAGE_SLEEP=true requires IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=true." >&2
        exit 1
    fi
    for base_url in "${VERIFIER_BASE_URL_ARRAY[@]}"; do
        IF_LLM_VERIFIER_CONTROL_URL="${base_url}" \
            bash "${SCRIPT_DIR}/control_qwen3_30ba3b_verifier.sh" sleep "${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    done
fi
########################### end verifier servers ###########################

########################### actor training ###########################
REWARD_MANAGER_PATH="${VERL_DIR}/if_rlvr/if_llm_verifier_reward_manager.py"

OVERRIDES=(
    "+if_ppl_prefix_mode=${IF_PPL_PREFIX_MODE}"
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
    trainer.default_local_dir="${CKPT_DIR}"
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]"
    "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=${HF_HOME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=${HF_HUB_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_APPLY_ENABLE_THINKING_KWARG=\"${IF_APPLY_ENABLE_THINKING_KWARG}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_ALLOW_MISSING_THINK_FINAL_ANSWER=\"${IF_ALLOW_MISSING_THINK_FINAL_ANSWER}\""
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
bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" "${OVERRIDES[@]}" "$@"
status=$?
set -e
exit "${status}"
