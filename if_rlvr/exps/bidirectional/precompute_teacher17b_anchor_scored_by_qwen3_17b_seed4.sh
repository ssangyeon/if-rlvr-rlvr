#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
WORKSPACE_DIR=$(cd -- "${VERL_DIR}/.." && pwd)

# Keep this entrypoint self-contained: use the repository's verl environment,
# the local project cache, the already-populated NLTK data, and this host's GPUs.
VERL_ENV_BIN=${VERL_ENV_BIN:-${WORKSPACE_DIR}/.miniforge3/envs/verl/bin}
if [[ ! -x "${VERL_ENV_BIN}/python3" ]]; then
    echo "verl Python not found: ${VERL_ENV_BIN}/python3" >&2
    exit 1
fi
export PATH="${VERL_ENV_BIN}:${PATH}"
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${WORKSPACE_DIR}/IFBench/.nltk_data}
export GPU_SET=${GPU_SET:-0,1,2,3}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

export IF_DATA_SEED=4
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher17b_nonreason_train_seed4_val512_scored_by_qwen3_17b.json}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_17b_precompute_teacher17b_nonreason_anchor_scored_by_qwen3_17b_seed4}

exec bash "${SCRIPT_DIR}/precompute_teacher17b_anchor_scored_by_qwen3_17b.sh" "$@"
