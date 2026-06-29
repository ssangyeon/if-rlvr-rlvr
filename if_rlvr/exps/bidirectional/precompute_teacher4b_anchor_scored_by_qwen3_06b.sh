#!/usr/bin/env bash
# Rescore 4B non-reasoning anchor answers with the Qwen3-0.6B reference policy.
#
# Intended use:
#   0.6B reasoning policy + 4B non-reasoning final-answer anchor
#
# This uses GPUs 0,1,2,3 by default, leaving GPUs 4,5,6,7 for the current run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CACHE_ROOT=${CACHE_ROOT:-/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr/.cache}
INPUT_CACHE=${INPUT_CACHE:-${CACHE_ROOT}/if_ref_anchor_qwen3_4b_const1_train_seed1_val512_thinkfalse.json}
OUTPUT_CACHE=${OUTPUT_CACHE:-${CACHE_ROOT}/if_ref_anchor_teacher4b_nonreason_train_seed1_val512_scored_by_qwen3_06b_thinktrue.json}

SCORER_MODEL=${SCORER_MODEL:-Qwen/Qwen3-0.6B}
ANCHOR_ANSWER_MODEL=${ANCHOR_ANSWER_MODEL:-Qwen/Qwen3-4B}
NUM_WORKERS=${NUM_WORKERS:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
DEFAULT_PYTHON_BIN=/NHNHOME/26msit001_A/IFIF/.miniforge3/envs/verl/bin/python3.12
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
        PYTHON_BIN=${DEFAULT_PYTHON_BIN}
    else
        PYTHON_BIN=python3
    fi
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "transformers", "datasets") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + ". Activate the verl training environment first, or set PYTHON_BIN=/path/to/env/bin/python."
    )
print(f"[env] using python: {sys.executable}")
PY

"${PYTHON_BIN}" "${REPO_DIR}/if_rlvr/precompute_anchor_cache_scored_by_ref.py" \
    --input-cache "${INPUT_CACHE}" \
    --output-cache "${OUTPUT_CACHE}" \
    --scorer-model "${SCORER_MODEL}" \
    --anchor-answer-model "${ANCHOR_ANSWER_MODEL}" \
    --num-workers "${NUM_WORKERS}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --max-prompt-length 16384 \
    --max-response-length 8192 \
    --response-length 8192 \
    --metadata-enable-thinking \
    "$@"
