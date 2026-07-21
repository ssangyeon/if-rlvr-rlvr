#!/usr/bin/env bash
# Run three GRPO experiments sequentially on GPUs 0,1,2,3:
#   1. Qwen3-4B t4b anchor, pyx=0.1, non-reasoning
#   2. Qwen3-1.7B CI95 outer anchor, pyx=0.1, non-reasoning
#   3. Qwen3-1.7B CI95 inner anchor, pyx=0.1, non-reasoning
# A failed run stops the sequence before the next experiment starts.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
WORKSPACE_DIR=$(cd -- "${VERL_DIR}/.." && pwd)

export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
export VERL_ENV_BIN=${VERL_ENV_BIN:-${WORKSPACE_DIR}/.miniforge3/envs/verl/bin}
export PATH="${VERL_ENV_BIN}:${PATH}"
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${WORKSPACE_DIR}/IFBench/.nltk_data}
export NLTK_DATA=${NLTK_DATA:-${NLTK_DATA_DIR}}
export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-0}

if [[ ! -x "${VERL_ENV_BIN}/python3" ]]; then
    echo "verl Python not found: ${VERL_ENV_BIN}/python3" >&2
    exit 1
fi

if [[ ! -d "${NLTK_DATA_DIR}/tokenizers/punkt_tab/english" ]]; then
    echo "NLTK punkt_tab is missing: ${NLTK_DATA_DIR}/tokenizers/punkt_tab/english" >&2
    exit 1
fi

bash "${SCRIPT_DIR}/qwen3_17b_t17b_anchor_pyx01_nonreason_ci95_inner.sh"
bash "${SCRIPT_DIR}/qwen3_17b_llmverifier_qwen3_1_7b_bonus01_nonreason_t8.sh"
bash "${SCRIPT_DIR}/qwen3_17b_llmverifier_qwen3_30ba3b_bonus01_nonreason_t7.sh"
