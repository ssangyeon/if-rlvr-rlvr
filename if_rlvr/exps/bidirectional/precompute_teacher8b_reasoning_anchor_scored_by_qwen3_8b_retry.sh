#!/usr/bin/env bash
# Generate Qwen3-8B reasoning-on anchor answers and score only the final answer
# tokens with the same Qwen3-8B reference policy. Incomplete samples are retried
# in later passes until the cache is complete.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CACHE_ROOT=${CACHE_ROOT:-/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr/.cache}

export GPU_SET=${GPU_SET:-4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-8B}
export ENABLE_THINKING=${ENABLE_THINKING:-true}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-true}
export IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY=${IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY:-true}
export IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES=${IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES:-0}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-512}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-98304}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-98304}
export ROLLOUT_N=${ROLLOUT_N:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
export IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE=${IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE:-1024}
export IF_REF_ANCHOR_CACHE_SAVE_INTERVAL=${IF_REF_ANCHOR_CACHE_SAVE_INTERVAL:-1024}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}
export IF_DATASET_HF=${IF_DATASET_HF:-allenai/IF_multi_constraints_upto5}
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher8b_reasoning_train_seed1_scored_by_qwen3_8b.json}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-false}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}

export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.5}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_precompute_teacher8b_reasoning_anchor_scored_by_qwen3_8b_finalonly_retry}

cache_summary() {
    python3 - "${IF_REF_ANCHOR_CACHE_PATH}" "${IF_DATASET_HF}" "${IF_VAL_SIZE}" <<'PY'
import json
import math
import os
import sys

path, dataset_name, val_size_s = sys.argv[1], sys.argv[2], sys.argv[3]
val_size = int(val_size_s)

expected = os.getenv("IF_REF_ANCHOR_EXPECTED_TOTAL", "").strip()
if expected:
    total = int(expected)
else:
    import datasets

    total = max(len(datasets.load_dataset(dataset_name, split="train")) - val_size, 0)

items = {}
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f).get("items", {})

complete = 0
for item in items.values():
    try:
        ref0_count = int(item.get("ref0_token_count", 0) or 0)
        ref1_count = int(item.get("ref1_token_count", 0) or 0)
        ref0_nll = float(item.get("ref0_nll", float("inf")))
        ref1_nll = float(item.get("ref1_nll", float("inf")))
    except (TypeError, ValueError):
        continue
    if (
        item.get("y0")
        and item.get("y1")
        and ref0_count > 0
        and ref1_count > 0
        and math.isfinite(ref0_nll)
        and math.isfinite(ref1_nll)
    ):
        complete += 1

missing = max(total - complete, 0)
print(f"complete={complete} total={total} missing={missing} path={path}")
sys.exit(0 if complete >= total else 1)
PY
}

try_idx=1
max_tries=${IF_REF_ANCHOR_MAX_TRIES:-0}

while true; do
    if summary=$(cache_summary); then
        echo "[IF ref anchor retry] cache complete: ${summary}"
        break
    else
        echo "[IF ref anchor retry] cache incomplete before try ${try_idx}: ${summary:-unavailable}"
    fi

    if (( max_tries > 0 && try_idx > max_tries )); then
        echo "[IF ref anchor retry] reached IF_REF_ANCHOR_MAX_TRIES=${max_tries}" >&2
        exit 1
    fi

    echo "[IF ref anchor retry] starting try ${try_idx}"
    set +e
    bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" \
        +if_ref_anchor_precompute_only=true \
        +if_ref_anchor_cache_save_interval="${IF_REF_ANCHOR_CACHE_SAVE_INTERVAL}" \
        actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
        trainer.logger='["console"]' \
        "$@"
    run_status=$?
    set -e

    if summary=$(cache_summary); then
        echo "[IF ref anchor retry] cache complete after try ${try_idx}: ${summary}"
        break
    fi

    echo "[IF ref anchor retry] cache still incomplete after try ${try_idx}: ${summary:-unavailable}"
    if (( run_status != 0 )); then
        echo "[IF ref anchor retry] try ${try_idx} exited with status ${run_status}; continuing with saved complete samples"
    fi
    try_idx=$((try_idx + 1))
done
