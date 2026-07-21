#!/usr/bin/env bash
# Resume the reward-cost benchmark from an already completed rollout-skip dump.

set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 STEP_DIR CORPUS_JSONL OUTPUT_DIR" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STEP_DIR=$1
CORPUS=$2
OUTPUT_ROOT=$3

PYTHON=${PYTHON:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/.miniforge3/envs/verl/bin/python}
VERL_DIR=${VERL_DIR:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr}
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NLTK_DATA=${NLTK_DATA:-/NHNHOME/WORKSPACE/26msit001_A/IFIF/IFBench/.nltk_data}

for required in new_batch.dp gen_batch.dp; do
    if [[ ! -s "${STEP_DIR}/${required}" ]]; then
        echo "ERROR: missing or empty ${STEP_DIR}/${required}" >&2
        exit 2
    fi
done

mkdir -p "$(dirname -- "${CORPUS}")" "${OUTPUT_ROOT}"

LOCAL_TOKENIZER_ARGS=()
if [[ "${LOCAL_FILES_ONLY:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    LOCAL_TOKENIZER_ARGS+=(--local-files-only)
fi

if [[ -s "${CORPUS}" ]]; then
    echo "[resume] validating existing corpus: ${CORPUS}"
    "${PYTHON}" "${SCRIPT_DIR}/benchmark_reward_cost.py" validate \
        --input "${CORPUS}" \
        --expected-responses 8192 \
        --expected-prompts 1024 \
        --rollouts-per-prompt 8
else
    echo "[resume] exporting rollout dump: ${STEP_DIR}"
    "${PYTHON}" "${SCRIPT_DIR}/prepare_rollout_step.py" \
        "${STEP_DIR}" \
        --output "${CORPUS}" \
        --tokenizer Qwen/Qwen3-4B \
        --max-prompt-length 16384 \
        --nltk-data "${NLTK_DATA}" \
        --expected-responses 8192 \
        --expected-prompts 1024 \
        --rollouts-per-prompt 8 \
        --require-think-end \
        "${LOCAL_TOKENIZER_ARGS[@]}"
fi

echo "[resume] starting four reward-cost measurements"
bash "${SCRIPT_DIR}/run_reward_cost_benchmark.sh" "${CORPUS}" "${OUTPUT_ROOT}"

echo "[resume] report: ${OUTPUT_ROOT}/report/report.md"
