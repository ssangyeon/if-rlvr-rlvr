#!/usr/bin/env bash
# Generate one run-specific missing-anchor set with four full Qwen3-4B vLLM replicas.
#
# This is deliberately a one-attempt launcher:
#   - no model-level empty-response retries;
#   - no hygiene regeneration passes;
#   - no launcher or campaign retry loop.
#
# The input batch is dynamically data-parallel across physical GPUs 4,5,6,7.
# tensor_model_parallel_size=1 means every GPU hosts one complete vLLM model.
set -Eeuo pipefail
umask 077

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd -- "${HERE}/../../.." && pwd)
IFIF_ROOT=$(cd -- "${VERL_DIR}/.." && pwd)
PYTHON_BIN=${VERL_ENV_PYTHON:-${IFIF_ROOT}/.miniforge3/envs/verl/bin/python}
COMMON_LAUNCHER="${VERL_DIR}/if_rlvr/exps/bidirectional/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh"
PREPARE_TOOL="${HERE}/prepare_anchor_views.py"
RUNTIME_ROOT=${SUBSET20K_RUNTIME_ROOT:-${VERL_DIR}/.agent_runtime/subset20k}
RAY_TEMP_PARENT=${SUBSET20K_RAY_TEMP_PARENT:-${VERL_DIR}/.rt20k}

RUN=${1:-}
case "${RUN}" in
    run1)
        EXPECTED_ROWS=2664
        SAMPLE_SEED=11000
        AGENT_WORKERS=72
        PRECOMPUTE_BATCH_ROWS=288
        PORT_BASE=26000
        ;;
    run2)
        EXPECTED_ROWS=1059
        SAMPLE_SEED=22000
        # 3 workers would expose at most three concurrent requests across four
        # replicas. 106 workers with 318-row batches keeps all GPUs fed; only
        # the final 105-row batch receives one throw-away padding request.
        AGENT_WORKERS=106
        PRECOMPUTE_BATCH_ROWS=318
        PORT_BASE=30000
        ;;
    run3)
        EXPECTED_ROWS=1066
        SAMPLE_SEED=33000
        AGENT_WORKERS=82
        PRECOMPUTE_BATCH_ROWS=328
        PORT_BASE=34000
        ;;
    *)
        echo "usage: $0 run1|run2|run3" >&2
        exit 2
        ;;
esac

MANIFEST="${HERE}/generation_manifests/${RUN}/missing_indices.json"
OUTPUT_CACHE="${RUNTIME_ROOT}/generated_runs/${RUN}.missing.cache.json"
[[ -x "${PYTHON_BIN}" ]] || { echo "missing executable Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -s "${COMMON_LAUNCHER}" ]] || { echo "missing common launcher: ${COMMON_LAUNCHER}" >&2; exit 2; }
[[ -s "${MANIFEST}" ]] || { echo "missing row manifest: ${MANIFEST}" >&2; exit 2; }

export PATH="$(dirname -- "${PYTHON_BIN}")":${PATH}
ACTUAL_ROWS=$(
    "${PYTHON_BIN}" -c \
        'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["indices"]))' \
        "${MANIFEST}"
)
[[ "${ACTUAL_ROWS}" == "${EXPECTED_ROWS}" ]] || {
    echo "${RUN}: manifest row count changed: expected ${EXPECTED_ROWS}, got ${ACTUAL_ROWS}" >&2
    exit 2
}
(( PRECOMPUTE_BATCH_ROWS % AGENT_WORKERS == 0 )) || {
    echo "${RUN}: precompute batch would cause hidden request padding" >&2
    exit 2
}
FINAL_ROWS=$((EXPECTED_ROWS % PRECOMPUTE_BATCH_ROWS))
FINAL_PADDING=0
if (( FINAL_ROWS > 0 )); then
    FINAL_PADDING=$(( (AGENT_WORKERS - FINAL_ROWS % AGENT_WORKERS) % AGENT_WORKERS ))
fi
(( FINAL_PADDING <= 1 )) || {
    echo "${RUN}: final precompute batch would add ${FINAL_PADDING} padding requests" >&2
    exit 2
}

if [[ "${SUBSET20K_PLAN_VALIDATED:-0}" != 1 ]]; then
    "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" validate-plan >/dev/null
fi
mkdir -p "$(dirname -- "${OUTPUT_CACHE}")" "${RUNTIME_ROOT}/logs" "${RAY_TEMP_PARENT}"

if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
    validate-generated-run --run "${RUN}" >/dev/null 2>&1; then
    echo "[subset20k] ${RUN} is already complete and strictly validated; no generation submitted."
    exit 0
fi
if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
    audit-generated-run --run "${RUN}" >/dev/null 2>&1; then
    echo "[subset20k] ${RUN} already records every one-shot attempt; no retry submitted."
    exit 0
fi
if [[ -e "${OUTPUT_CACHE}" ]]; then
    echo "[subset20k] refusing structurally invalid or partial output: ${OUTPUT_CACHE}" >&2
    echo "[subset20k] preserve and inspect it, then explicitly archive it before a fresh attempt." >&2
    exit 3
fi

# One isolated Ray control plane creates four rollout replicas. Requests from the
# full run-specific batch are asynchronously routed to the least-loaded replica.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DEVICE=gpu
export INFER_BACKEND=vllm
export NNODES=1
export MACHINE=
export GPU_SET=4,5,6,7
export NGPUS_PER_NODE=4
export ROLLOUT_TP=1
export IF_RLVR_ISOLATE_RAY=1
unset IF_RLVR_CONCURRENT_ENV_SH
export RUN_SLOT=12
export IF_RLVR_PORT_BASE="${PORT_BASE}"
export IF_RLVR_RUN_ID="subset20k_${RUN}_oneshot_$$"
export RAY_TMPDIR=$(mktemp -d "${RAY_TEMP_PARENT}/${RUN}.XXXXXX")
export RUN_TMPDIR="${RAY_TMPDIR}/work"
export IF_RLVR_OBJECT_STORE_BYTES=68719476736
export IF_RLVR_NUM_CPUS=176
mkdir -p "${RUN_TMPDIR}"

export HF_HOME=${HF_HOME:-/data/IFIF/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}

export MODEL_PATH=Qwen/Qwen3-4B
export IF_REF_VLLM_MODEL=Qwen/Qwen3-4B
export IF_REF_VLLM_BASE_URL=
export IF_DATASET_HF=allenai/IF_multi_constraints_upto5
export IF_PPL_PREFIX_MODE=standard
export IF_REF_ANCHOR_PRECOMPUTE=true
export IF_REF_POLICY_ANCHOR_PPL=true
export IF_REF_PPL_BASELINE=0
export IF_REF_PPL_ANCHOR=0
export ENABLE_THINKING=true
export IF_APPLY_ENABLE_THINKING_KWARG=true
export IF_REQUIRE_THINK_END_FOR_REWARD=true
export IF_ALLOW_MISSING_THINK_FINAL_ANSWER=false
export IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY=true
export IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES=0
export IF_ANCHOR_PRECOMPUTE_MIN_TOKENS=10

export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=0.95
export ROLLOUT_TOP_K=20
export PRESENCE_PENALTY=0.0
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=32768
export ROLLOUT_N=1

export TRAIN_BATCH_SIZE="${EXPECTED_ROWS}"
export PPO_MINI_BATCH_SIZE="${EXPECTED_ROWS}"
export PPO_MAX_TOKEN_LEN_PER_GPU=98304
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=98304
export ROLLOUT_GPU_MEM_UTIL=0.55
export AGENT_NUM_WORKERS="${AGENT_WORKERS}"
export DATA_PROCESSOR_CPU_COUNT=32
export IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_ROWS}"
export IF_REF_ANCHOR_CACHE_SAVE_INTERVAL=64

export IF_DATA_SEED=1
export IF_VAL_SIZE=512
export IF_TRAIN_INDEX_FILTER="${MANIFEST}"
export IF_REF_ANCHOR_EXPECTED_TOTAL="${EXPECTED_ROWS}"
export IF_REF_ANCHOR_CACHE_PATH="${OUTPUT_CACHE}"
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=true
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=false
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=false
export IF_MAX_RETRIES=1
export RESUME_MODE=disable
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_EPOCHS=1
export WANDB_MODE=disabled

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "[subset20k] run=${RUN} rows=${EXPECTED_ROWS} anchor_draws=${EXPECTED_ROWS}"
echo "[subset20k] full_completions=$((2 * EXPECTED_ROWS)) scoring_calls=$((2 * EXPECTED_ROWS))"
echo "[subset20k] GPUs=4,5,6,7 replicas=4 TP=1 routing=least-inflight"
echo "[subset20k] t=1.0 p=0.95 k=20 pp=0 prompt=2048 completion=32768"
echo "[subset20k] agent_workers=${AGENT_WORKERS} rows_per_worker=$((EXPECTED_ROWS / AGENT_WORKERS))"
echo "[subset20k] final_batch_padding=${FINAL_PADDING} (discarded before persistence)"
echo "[subset20k] persistence_batch=${PRECOMPUTE_BATCH_ROWS} vllm_gpu_memory_utilization=0.55"
echo "[subset20k] retry policy: empty=0 launcher=0 campaign=0"

set +e
bash "${COMMON_LAUNCHER}" \
    +if_ref_anchor_precompute_only=true \
    +if_ref_anchor_cache_save_interval=64 \
    actor_rollout_ref.rollout.response_length=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.max_model_len=40960 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.logger='["console"]' \
    trainer.resume_mode=disable \
    +ray_kwargs.ray_init.num_cpus=176 \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${SAMPLE_SEED}"
LAUNCH_RC=$?
set -e

if (( LAUNCH_RC != 0 )); then
    if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
        audit-generated-run --run "${RUN}" >/dev/null 2>&1; then
        echo "[subset20k] launcher rc=${LAUNCH_RC}, but every declared attempt is recorded; continuing." >&2
    else
        echo "[subset20k] ${RUN} failed on its sole launch attempt (rc=${LAUNCH_RC})." >&2
        echo "[subset20k] no automatic retry will be made; any partial cache is preserved." >&2
        exit "${LAUNCH_RC}"
    fi
fi

"${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
    audit-generated-run --run "${RUN}"
if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
    validate-generated-run --run "${RUN}" >/dev/null 2>&1; then
    echo "[subset20k] ${RUN} completed and every row is train-ready."
else
    echo "[subset20k] ${RUN} recorded every one-shot attempt, but some rows are not train-ready." >&2
    echo "[subset20k] continuing to the next run without retrying this one." >&2
fi
