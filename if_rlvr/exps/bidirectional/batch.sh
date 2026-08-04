#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this script so the same batch file works from either
# the T_A workspace or a container-mounted A workspace.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IFIF_ROOT="$(dirname "${VERL_DIR}")"
CONDA_PREFIX="${IFIF_ROOT}/.miniforge3/envs/verl"

cd "${VERL_DIR}"

# 복사된 Conda 환경을 수동 활성화
export CONDA_PREFIX
export CONDA_DEFAULT_ENV=verl
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
hash -r

# 프로젝트 경로
export VERL_DIR
export CACHE_ROOT="${VERL_DIR}/.cache"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# Always keep runtime data under the workspace that contains this batch file.
# The copied Conda environment uses certifi's CA bundle; exporting it prevents
# NLTK downloads from failing on hosts without a configured system CA bundle.
export NLTK_DATA_DIR="${IFIF_ROOT}/IFBench/.nltk_data"
export NLTK_DATA="${NLTK_DATA_DIR}"
mkdir -p "${NLTK_DATA_DIR}"

CERTIFI_CA="$(${CONDA_PREFIX}/bin/python -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE="${CERTIFI_CA}"
export REQUESTS_CA_BUNDLE="${CERTIFI_CA}"

"${CONDA_PREFIX}/bin/python" - <<'PY'
import os
import nltk

data_dir = os.environ["NLTK_DATA"]
if data_dir not in nltk.data.path:
    nltk.data.path.insert(0, data_dir)

for package, resource in {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "punkt": "tokenizers/punkt/english.pickle",
}.items():
    try:
        nltk.data.find(resource)
    except LookupError:
        print(f"[batch] downloading NLTK {package} to {data_dir}")
        nltk.download(package, download_dir=data_dir, raise_on_error=True)
        nltk.data.find(resource)
PY

# GPU 0,1,2,3 사용
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPU_SET=0,1,2,3
export NGPUS_PER_NODE=4

# Shared verifier inference settings.
export IF_LLM_VERIFIER_REWARD_WORKERS=32
export IF_LLM_VERIFIER_MAX_TOKENS=1024
export IF_LLM_VERIFIER_MAX_RETRIES=0
export IF_LLM_VERIFIER_GPU_MEM_UTIL=0.80
export IF_LLM_VERIFIER_SERVER_RESTART_ATTEMPTS=1

# Leave enough headroom for the larger per-GPU FSDP shard on 2-GPU runs.
export ROLLOUT_GPU_MEM_UTIL=0.80

# W&B 인증은 배치 시스템의 secret/environment variable로 주입
: "${WANDB_API_KEY:?WANDB_API_KEY가 설정되지 않았습니다}"
export WANDB_API_KEY
export WANDB_ENTITY=ifif

# OpenAI API는 이번 로컬 Qwen verifier 학습에는 사용하지 않음
# 필요한 경우에만 배치 시스템 secret으로 OPENAI_API_KEY를 주입
# : "${OPENAI_API_KEY:?OPENAI_API_KEY가 설정되지 않았습니다}"

# 캐시 확인
REASONING_ANCHOR_CACHE="${CACHE_ROOT}/if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.json"
NONREASON_ANCHOR_CACHE="${CACHE_ROOT}/if_ref_anchor_teacher4b_nonreason_train_seed1_val512_scored_by_qwen3_4b.json"

if [[ ! -s "${REASONING_ANCHOR_CACHE}" ]]; then
    echo "ERROR: reasoning anchor cache가 없습니다: ${REASONING_ANCHOR_CACHE}" >&2
    exit 1
fi
if [[ ! -s "${NONREASON_ANCHOR_CACHE}" ]]; then
    echo "ERROR: non-reasoning anchor cache가 없습니다: ${NONREASON_ANCHOR_CACHE}" >&2
    exit 1
fi

# Resolve Qwen3-4B to a complete local Hub snapshot.  Four rollout servers
# starting from the repository ID at once can trigger the Hub's per-IP 429
# limit even when the weights are already cached.
HF_HUB_CACHE_ROOT="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
shopt -s nullglob
QWEN4_SNAPSHOTS=("${HF_HUB_CACHE_ROOT}/models--Qwen--Qwen3-4B/snapshots/"*)
shopt -u nullglob

QWEN4_LOCAL_MODEL=""
for snapshot in "${QWEN4_SNAPSHOTS[@]}"; do
    if [[ -s "${snapshot}/config.json" && -s "${snapshot}/model.safetensors.index.json" ]]; then
        QWEN4_LOCAL_MODEL="${snapshot}"
    fi
done

if [[ -z "${QWEN4_LOCAL_MODEL}" ]]; then
    echo "ERROR: complete local Qwen3-4B snapshot not found under ${HF_HUB_CACHE_ROOT}" >&2
    exit 1
fi

echo "Using local Qwen3-4B snapshot: ${QWEN4_LOCAL_MODEL}"
export MODEL_PATH="${QWEN4_LOCAL_MODEL}"
export IF_REF_VLLM_MODEL="${QWEN4_LOCAL_MODEL}"

# 실행 충돌 방지용 포트
export RUN_SLOT="${RUN_SLOT:-1}"
export IF_RLVR_PORT_BASE="${IF_RLVR_PORT_BASE:-22000}"
export VLLM_MASTER_PORT_BASE="${VLLM_MASTER_PORT_BASE:-40000}"

# Run the Qwen verifier experiment first, followed by the three Llama jobs.
# Scope the Qwen anchor cache to that child process.

IF_REF_ANCHOR_CACHE_PATH="${REASONING_ANCHOR_CACHE}" \
    bash if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_qwen3_30ba3b_bonus01_reasoning.sh

# Llama 3.1 is a gated, non-reasoning model. Override the Qwen snapshot exported
# above for each child process; the Llama common launcher resolves this Hub ID
# once to a complete local snapshot before starting Ray/vLLM.
LLAMA31_MODEL="meta-llama/Llama-3.1-8B-Instruct"

MODEL_PATH="${LLAMA31_MODEL}" IF_REF_VLLM_MODEL="${LLAMA31_MODEL}" \
    bash if_rlvr/exps/bidirectional/precompute_teacher_llama31_8b_instruct_anchor_scored_by_llama31_8b.sh

MODEL_PATH="${LLAMA31_MODEL}" IF_REF_VLLM_MODEL="${LLAMA31_MODEL}" \
    bash if_rlvr/exps/bidirectional/llama31_8b_constraint_only_nonreason.sh

MODEL_PATH="${LLAMA31_MODEL}" IF_REF_VLLM_MODEL="${LLAMA31_MODEL}" \
    bash if_rlvr/exps/bidirectional/llama31_8b_t8b_anchor_pyx01_nonreason.sh
