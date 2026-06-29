#!/usr/bin/env bash
# Gemma-4-E2B-it non-reasoning policy with Gemma-4-E2B-it non-reasoning final-answer anchors,
# rescored by the Gemma-4-E2B-it reference policy using non-reasoning PPL prefixes.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BIDIRECTIONAL_SCRIPT_DIR="${SCRIPT_DIR}/../bidirectional"
VERL_DIR=${VERL_DIR:-/NHNHOME/26msit001_A/IFIF/if-rlvr}
CACHE_ROOT=${CACHE_ROOT:-/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr/.cache}

cd "${VERL_DIR}"

GEMMA_CONDA_PREFIX=${GEMMA_CONDA_PREFIX:-/NHNHOME/26msit001_A/IFIF/.miniforge3/envs/verl_gemma}
if [[ "${IF_GEMMA_USE_CONDA_ENV:-true}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && -x "${GEMMA_CONDA_PREFIX}/bin/python3" ]]; then
    export CONDA_PREFIX="${GEMMA_CONDA_PREFIX}"
    export CONDA_DEFAULT_ENV=verl_gemma
    export PATH="${GEMMA_CONDA_PREFIX}/bin:${PATH}"
fi

export GPU_SET=${GPU_SET:-4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export RUN_SLOT=${RUN_SLOT:-1}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-40000}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export MODEL_PATH=${MODEL_PATH:-google/gemma-4-E2B-it}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-google/gemma-4-E2B-it}
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-false}
export IF_ALLOW_MISSING_THINK_FINAL_ANSWER=${IF_ALLOW_MISSING_THINK_FINAL_ANSWER:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
export ROLLOUT_CUDAGRAPH_MODE=${ROLLOUT_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
export ROLLOUT_KV_CACHE_MEMORY_BYTES=${ROLLOUT_KV_CACHE_MEMORY_BYTES:-}
export ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
export ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-False}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
export SAVE_FREQ=${SAVE_FREQ:-91}
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_gemma4_e2b_it_nonreason_train_seed1_scored_by_gemma4_e2b_it.json}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}


if [[ "${IF_GEMMA4_PREFLIGHT:-true}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    python3 "${SCRIPT_DIR}/_gemma4_runtime_check.py" "${MODEL_PATH}"
    if [[ "${IF_REF_VLLM_MODEL}" != "${MODEL_PATH}" ]]; then
        python3 "${SCRIPT_DIR}/_gemma4_runtime_check.py" "${IF_REF_VLLM_MODEL}"
    fi
fi

if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "Missing anchor cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    echo "Run ${SCRIPT_DIR}/precompute_gemma4_e2b_anchor_scored_by_gemma4_e2b.sh first." >&2
    exit 1
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_e2b_it_grpo_nonthink_pyx01_tgemma4e2bitanchor_sgemma4e2bit_b1024_c1}

GEMMA_VLLM_OVERRIDES=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config={cudagraph_mode:${ROLLOUT_CUDAGRAPH_MODE}}"
)
if [[ -n "${ROLLOUT_KV_CACHE_MEMORY_BYTES}" ]]; then
    GEMMA_VLLM_OVERRIDES=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=${ROLLOUT_KV_CACHE_MEMORY_BYTES}"
        "${GEMMA_VLLM_OVERRIDES[@]}"
    )
fi

exec bash "${BIDIRECTIONAL_SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
    "${GEMMA_VLLM_OVERRIDES[@]}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY}" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_ALLOW_MISSING_THINK_FINAL_ANSWER=${IF_ALLOW_MISSING_THINK_FINAL_ANSWER}" \
    "$@"
