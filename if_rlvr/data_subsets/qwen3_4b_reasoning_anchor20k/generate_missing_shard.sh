#!/usr/bin/env bash
# Generate one exact missing-row data shard with one Qwen3-4B model replica.
set -Eeuo pipefail
umask 077

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd -- "${HERE}/../../.." && pwd)
IFIF_ROOT=$(cd -- "${VERL_DIR}/.." && pwd)
PYTHON_BIN=${VERL_ENV_PYTHON:-${IFIF_ROOT}/.miniforge3/envs/verl/bin/python}
PRECOMPUTE_SCRIPT="${VERL_DIR}/if_rlvr/exps/bidirectional/precompute_teacher4b_reasoning_anchor_v2_scored_by_qwen3_4b.sh"
PREPARE_TOOL="${HERE}/prepare_anchor_views.py"
RUNTIME_ROOT=${SUBSET20K_RUNTIME_ROOT:-${VERL_DIR}/.agent_runtime/subset20k}

RUN=${1:?usage: generate_missing_shard.sh run1|run2|run3 shard0..3 physical_gpu}
SHARD=${2:?usage: generate_missing_shard.sh run1|run2|run3 shard0..3 physical_gpu}
PHYSICAL_GPU=${3:?usage: generate_missing_shard.sh run1|run2|run3 shard0..3 physical_gpu}
[[ "${RUN}" =~ ^run[123]$ ]] || { echo "invalid run: ${RUN}" >&2; exit 2; }
[[ "${SHARD}" =~ ^[0-3]$ ]] || { echo "invalid shard: ${SHARD}" >&2; exit 2; }
[[ "${PHYSICAL_GPU}" =~ ^[4-7]$ ]] || { echo "physical GPU must be 4,5,6,7" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "training Python is missing: ${PYTHON_BIN}" >&2; exit 2; }
[[ -s "${PRECOMPUTE_SCRIPT}" ]] || { echo "precompute launcher is missing" >&2; exit 2; }

MANIFEST="${HERE}/generation_manifests/${RUN}/shard${SHARD}.indices.json"
[[ -s "${MANIFEST}" ]] || { echo "missing shard manifest: ${MANIFEST}" >&2; exit 2; }
readarray -t SHARD_INFO < <("${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload["row_count"])
print(payload["output_cache"])
print(payload["generation"]["sample_seed"])
PY
)
ROW_COUNT=${SHARD_INFO[0]}
OUTPUT_CACHE=${SHARD_INFO[1]}
SAMPLE_SEED=${SHARD_INFO[2]}
[[ "${ROW_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid shard row count" >&2; exit 2; }
mkdir -p "$(dirname -- "${OUTPUT_CACHE}")" "${RUNTIME_ROOT}/tmp/${RUN}_s${SHARD}"

if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
    validate-shard --run "${RUN}" --shard "${SHARD}" >/dev/null 2>&1; then
    echo "[subset20k-generation] ${RUN}/shard${SHARD} already complete and verified (${ROW_COUNT} rows)"
    exit 0
fi

# Four independent Ray/vLLM control planes: one complete model replica and one
# data shard per physical GPU. This is data parallelism, not model sharding.
PORT_BASE=$((50000 + SHARD * 3000))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GPU_SET="${PHYSICAL_GPU}"
export NGPUS_PER_NODE=1
export ROLLOUT_TP=1
export RUN_SLOT=$((20 + SHARD))
export IF_RLVR_PORT_BASE="${PORT_BASE}"
export IF_RLVR_RUN_ID="subset20k_${RUN}_s${SHARD}_gpu${PHYSICAL_GPU}"
export RAY_TMPDIR="/tmp/subset20k_${RUN}_s${SHARD}"
export RUN_TMPDIR="${RUNTIME_ROOT}/tmp/${RUN}_s${SHARD}"
export IF_RLVR_OBJECT_STORE_BYTES=17179869184
export IF_RLVR_NUM_CPUS=40

export HF_HOME=${HF_HOME:-/data/IFIF/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}

export MODEL_PATH=Qwen/Qwen3-4B
export IF_REF_VLLM_MODEL=Qwen/Qwen3-4B
export ENABLE_THINKING=true
export IF_APPLY_ENABLE_THINKING_KWARG=true
export IF_REQUIRE_THINK_END_FOR_REWARD=true
export IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY=true
export IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES=2
export IF_ANCHOR_PRECOMPUTE_MIN_TOKENS=10

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=0.95
export ROLLOUT_TOP_K=20
export PRESENCE_PENALTY=0.0
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export ROLLOUT_N=2

export TRAIN_BATCH_SIZE="${ROW_COUNT}"
export PPO_MINI_BATCH_SIZE="${ROW_COUNT}"
export PPO_MAX_TOKEN_LEN_PER_GPU=98304
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304
export ROLLOUT_GPU_MEM_UTIL=0.7
export AGENT_NUM_WORKERS=32
export DATA_PROCESSOR_CPU_COUNT=8
export IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE="${ROW_COUNT}"
export IF_REF_ANCHOR_CACHE_SAVE_INTERVAL=8
export IF_ANCHOR_HYGIENE_PASSES=6

export IF_DATA_SEED=1
export IF_VAL_SIZE=512
export IF_TRAIN_INDEX_FILTER="${MANIFEST}"
export IF_REF_ANCHOR_EXPECTED_TOTAL="${ROW_COUNT}"
export IF_REF_ANCHOR_CACHE_PATH="${OUTPUT_CACHE}"
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=false
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=false
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=false
export IF_MAX_RETRIES=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[subset20k-generation] ${RUN}/shard${SHARD} attempt=${attempt} rows=${ROW_COUNT} gpu=${PHYSICAL_GPU} seed=${SAMPLE_SEED}"
    set +e
    bash "${PRECOMPUTE_SCRIPT}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.max_model_len=40960 \
        actor_rollout_ref.rollout.max_num_seqs=128 \
        actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        "+ray_kwargs.ray_init.num_cpus=${IF_RLVR_NUM_CPUS}" \
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${SAMPLE_SEED}"
    launch_rc=$?
    set -e
    if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
        validate-shard --run "${RUN}" --shard "${SHARD}"; then
        echo "[subset20k-generation] ${RUN}/shard${SHARD} complete after attempt ${attempt}"
        exit 0
    fi
    echo "[subset20k-generation] ${RUN}/shard${SHARD} remains incomplete (launcher rc=${launch_rc}); retrying in 30s" >&2
    sleep 30
done
