#!/usr/bin/env bash
# GRPO | allenai/Llama-3.1-Tulu-3-8B-DPO (non-reasoning)
#   + validated one-sample x / x+c anchor reward (teacher == policy init)
#   + Qwen/Qwen3-30B-A3B LLM-verifier fallback bonus at score threshold 7
#
# This is the Tulu 3 counterpart of
#   qwen3_4b_t4b_anchor_pyx01_llmverifier_gptoss120b_bonus01_reasoning.sh
# using the anchor cache and paper hyper-parameters of
#   llama31_tulu3_8b_dpo_anchor_grpo_nonreason.sh
# sized for one node of 8x H100 80GB.
#
# ---------------------------------------------------------------------------
# Per-rollout reward order (if_llm_verifier_anchor_fallback_only=true)
# ---------------------------------------------------------------------------
#   1. policy rollout on GPUs 0-7 (vLLM, TP=1 per GPU)
#   2. IFEval constraint reward; rows with constraint=0 stop here
#   3. anchor NLL of the sampled answer is scored against the cached
#      [x-only anchor A, x+c anchor B] interval, for constraint-positive rows
#   4. rows inside [A, B] receive the +0.1 anchor bonus
#   5. every other constraint-positive row gets ONE Qwen3-30B-A3B judge call:
#        - judge score >= 7 and the lower gate had zeroed the row -> constraint
#          reward is restored and +0.1 is added
#        - judge score >= 7 otherwise -> constraint reward is preserved, +0.1
#      so the two positive bonuses can never stack to +0.2.
#   Validation stays constraint-only: the anchor cache holds train rows only.
#
# ---------------------------------------------------------------------------
# GPU plan - both models are TP=1 on every GPU, so no cross-GPU collectives
# ---------------------------------------------------------------------------
#   * 8 verifier replicas (one Qwen3-30B-A3B per GPU, 61 GB of bf16 weights)
#     are started first on empty GPUs, then put to sleep (level 1: weights go
#     to host RAM, KV cache is freed) before the trainer touches the GPUs.
#   * Training then owns the GPUs. The reward loop wakes the verifier only for
#     the fallback phase and sleeps it again in a `finally` block, so the 61 GB
#     of judge weights and the training footprint are never resident together.
#   * The policy rollout engine sleeps at level 2 right after generation, and
#     the actor keeps parameters/optimizer state on the host outside its own
#     compute windows (ACTOR_PARAM_OFFLOAD / ACTOR_OPTIMIZER_OFFLOAD). That is
#     what leaves ~70 GB free per GPU for the awake verifier; those two flags
#     only move memory, they do not change the optimization.
#
#   Per-GPU budget at 80 GB, from measurement on this node. The trainer is NOT
#   at ~0 when the judge wakes: the policy engine does drop to 4.1 GB after its
#   level-2 sleep, but the ref-policy log-prob pass that computes the anchor NLL
#   runs immediately before the judge phase and leaves ~10 GB in the caching
#   allocator, for a measured 13.9 GB residual at wake time.
#
#   That is why the judge runs TP=2 (4 replicas over 8 GPUs) rather than TP=1:
#     TP=1: 61.1 GB of bf16 weights + 13.9 residual leaves ~3 GB of KV, roughly
#           7 concurrent judges per replica - the judge becomes the bottleneck.
#     TP=2: 30.5 GB of weights per GPU, ~30 GB of KV per GPU (60 GB per
#           replica), fits with >3 GB of headroom, and halves the per-step
#           sleep/wake transfer (30.5 GB instead of 61.1 GB per GPU at the
#           ~1.8 GB/s the CuMemAllocator offload achieves).
#   Every GPU still runs both models; no GPU is dedicated to a single role, and
#   a replica's two ranks are an NVLink-local pair.
#
#   Note: IF_LLM_VERIFIER_SLEEP_LEVEL=2 is NOT a valid way to cut the transfer
#   cost. Level 2 discards the weights and expects an external weight update on
#   wake, which a standalone judge server has no way to provide.
#
# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
#   * wandb: trainer.logger=[console,wandb], ${WANDB_ENTITY}/${PROJECT_NAME}.
#   * Hugging Face Hub: `hf_model` is added to the actor checkpoint contents, so
#     every saved step also lands as a ready-to-load bf16 HF model. A watcher
#     (push_checkpoints_to_hf.py) uploads each completed step to
#     ${HF_CHECKPOINT_REPO} under `global_step_<N>/`, and a final sweep runs on
#     exit. Load one with:
#         AutoModelForCausalLM.from_pretrained(repo, subfolder="global_step_91")
#
# Usage:  bash llama31_tulu3_8b_dpo_anchor_pyx01_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh
# Any extra arguments are forwarded verbatim as Hydra overrides.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
DATA_ROOT=$(dirname -- "${VERL_DIR}")
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
cd "${VERL_DIR}"

########################### model / cache placement ###########################
# Tulu 3 8B (16 GB) plus Qwen3-30B-A3B (61 GB) need ~80 GB of HF cache. Small
# container-default cache mounts silently break the run at download time, so
# fall back to a roomier disk instead of failing 20 minutes in.
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
        echo "       Point HF_HOME at a disk that can hold Tulu-3-8B (16 GB) + Qwen3-30B-A3B (61 GB)." >&2
        exit 1
    fi
fi
export HF_HOME
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${CACHE_ROOT}"

########################### run isolation ###########################
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-tulu3_8b_dpo_anchor_q30ba3b_t7_slot${RUN_SLOT}}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-$((20000 + RUN_SLOT * 2000))}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-$((IF_RLVR_PORT_BASE + 200))}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

# Keep per-worker native thread pools small; Ray already creates many workers.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

########################### Tulu 3 runtime + defaults ###########################
# NOTE: everything in this block must be set BEFORE the shared file is sourced.
# That file exports these with its own `:-` defaults, so a later assignment here
# would silently lose to it.
#
# Whole-node run: claim all 8 GPUs before the shared file applies its 4-GPU default.
export GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Rollout engine sizing. Measured at 0.72: the rollout phase peaked at 73,727 of
# 81,559 MiB (90.4%), which left no room for keeping actor parameters and Adam
# state resident. 0.62 frees ~10 GiB so both offloads can stay off (see
# ACTOR_PARAM_OFFLOAD below), at the cost of ~70 rather than ~89 concurrent
# sequences per replica - rollout ran at 75-91% util and was not KV-starved.
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.72}

# The shared file expects the original cluster's conda/venv layout. Prefer the
# environment that is already active when it can run vLLM.
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

# tulu3_require_runtime only does `import vllm`. Every engine in this run - the
# policy rollout and all judge replicas - additionally imports the v1 worker
# stack, which pulls in numba via v1/spec_decode/ngram_proposer. A numba/numpy
# mismatch there kills each engine several minutes after launch, with the real
# cause buried in eight separate server logs. Surface it here in ten seconds.
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
# Paper/Tulu-specific: temperature 1.0, top_p 0.95, 2048-token responses,
# actor LR 5e-7. Batch 1024 x n=8 matches the sibling anchor comparison runs.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
# 93,993 cached rows / 1024 = 91 steps per epoch -> one checkpoint per epoch.
export SAVE_FREQ=${SAVE_FREQ:-91}
# 98304 is the B200 profile of the sibling scripts; 24576 is the 8B-on-80GB
# budget already used by if_rlvr/run_qwen3_8b_if_rlvr.sh.
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-32}
# ROLLOUT_GPU_MEM_UTIL is set above, before the shared file is sourced.

# Host-side parameter/optimizer residency. Pure memory placement - it does not
# change the optimization - but it is not free: with both enabled the run
# measured 52.2% mean GPU utilisation, 42.6% of samples below 20%, because every
# log-prob pass and the actor update pay a host round-trip. Per step that is
# ~24 GB of traffic for the fp32 parameters (3 round-trips) and ~16 GB for the
# Adam state (1), at roughly 2 GB/s.
#
# Keeping both resident costs 4,100 + 8,192 MiB per GPU, which is funded by the
# reduced ROLLOUT_GPU_MEM_UTIL above and IF_LLM_VERIFIER_GPU_MEM_UTIL below.
# If you raise either utilisation back up, turn these back on or the rollout
# phase will OOM.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}

########################### anchor reward ###########################
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_PPL_REWARD_STRATEGY=${IF_PPL_REWARD_STRATEGY:-anchor}
export IF_PPL_ANCHOR_REWARD_MODE=${IF_PPL_ANCHOR_REWARD_MODE:-both}
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-true}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-true}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}
# The published cache records the build host's tokenizer_class and its own
# train_sample_count, which no other machine can reproduce exactly. The full
# dict comparison is therefore replaced by the explicit semantic preflight
# below; the trainer still enforces ppl_prefix_mode / ppl_nll_scope itself.
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}

if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "[anchor] downloading hf://datasets/${TULU3_ANCHOR_HF_REPO}/${TULU3_ANCHOR_CACHE_FILENAME}"
    "${TULU3_PYTHON_BIN}" - <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

dest = os.environ["IF_REF_ANCHOR_CACHE_PATH"]
src = hf_hub_download(
    os.environ["TULU3_ANCHOR_HF_REPO"],
    filename=os.environ["TULU3_ANCHOR_CACHE_FILENAME"],
    repo_type="dataset",
)
os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
if os.path.realpath(src) != os.path.realpath(dest):
    shutil.copyfile(src, dest)
print(f"[anchor] downloaded {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
PY
fi
if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "ERROR: missing Tulu anchor cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    exit 1
fi

# Assert every cache field that changes what the anchors mean. Informational
# build-host fields (tokenizer_class, train_sample_count) are ignored on purpose.
"${TULU3_PYTHON_BIN}" - <<'PY'
import json
import os

path = os.environ["IF_REF_ANCHOR_CACHE_PATH"]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
metadata = payload.get("metadata", {})
expected = {
    "version": 3,
    "model_path": os.environ["MODEL_PATH"],
    "ppl_prefix_mode": os.environ["IF_PPL_PREFIX_MODE"],
    "ppl_nll_scope": "final_answer_tokens_only",
    "apply_chat_template_kwargs": {},
    "if_dataset_seed": int(os.environ["IF_DATA_SEED"]),
    "if_dataset_val_size": int(os.environ["IF_VAL_SIZE"]),
    "max_prompt_length": int(os.environ["MAX_PROMPT_LENGTH"]),
    "max_response_length": int(os.environ["MAX_RESPONSE_LENGTH"]),
    "response_length": int(os.environ["MAX_RESPONSE_LENGTH"]),
    "rollout_temperature": float(os.environ["ROLLOUT_TEMPERATURE"]),
    "rollout_top_p": float(os.environ["ROLLOUT_TOP_P"]),
    "rollout_top_k": -1,
}
mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
if mismatched:
    raise SystemExit(f"[anchor preflight] cache metadata mismatch (got, expected): {mismatched}")

items = payload.get("items", {})
complete = sum(
    1
    for item in items.values()
    if item.get("y0") and item.get("y1") and int(item.get("ref0_token_count", 0) or 0) > 0
    and int(item.get("ref1_token_count", 0) or 0) > 0
)
if complete == 0:
    raise SystemExit(f"[anchor preflight] no complete anchor rows in {path}")
print(
    f"[anchor preflight] OK: {complete}/{len(items)} complete rows; semantic metadata verified "
    "(tokenizer_class/train_sample_count ignored as build-host informational fields)"
)
PY

########################### LLM verifier ###########################
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-Qwen/Qwen3-30B-A3B}
export IF_LLM_VERIFIER_REVISION=${IF_LLM_VERIFIER_REVISION:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}
# Qwen3 is a hybrid-reasoning model; judge in non-thinking mode so the reply is
# just the {"Score": N} JSON object.
export IF_LLM_VERIFIER_ENABLE_THINKING=${IF_LLM_VERIFIER_ENABLE_THINKING:-false}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-${GPU_SET}}
# TP=2 -> 4 replicas over the 8 GPUs, each an NVLink-local pair. See the header:
# TP=1 does not leave a usable KV cache next to the trainer's measured residual.
export IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP:-1}
export IF_LLM_VERIFIER_HOST=${IF_LLM_VERIFIER_HOST:-127.0.0.1}
# The policy's own vLLM replicas reserve VLLM_MASTER_PORT_BASE + rank * 100, i.e.
# base+200 .. base+999 for 8 replicas. Start the judge servers above that block.
export IF_LLM_VERIFIER_PORT=${IF_LLM_VERIFIER_PORT:-$((IF_RLVR_PORT_BASE + 1200))}
export IF_LLM_VERIFIER_START_SERVER=${IF_LLM_VERIFIER_START_SERVER:-true}
export IF_LLM_VERIFIER_PYTHON=${IF_LLM_VERIFIER_PYTHON:-${TULU3_PYTHON_BIN}}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier}

# Wake/sleep alternation with the trainer.
export IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE:-true}
export IF_LLM_VERIFIER_MANAGE_SLEEP=${IF_LLM_VERIFIER_MANAGE_SLEEP:-true}
export IF_LLM_VERIFIER_SLEEP_LEVEL=${IF_LLM_VERIFIER_SLEEP_LEVEL:-1}
export IF_LLM_VERIFIER_DEV_MODE=${IF_LLM_VERIFIER_DEV_MODE:-1}
export IF_LLM_VERIFIER_CONTROL_TIMEOUT=${IF_LLM_VERIFIER_CONTROL_TIMEOUT:-600}
export IF_LLM_VERIFIER_WAIT_TIMEOUT=${IF_LLM_VERIFIER_WAIT_TIMEOUT:-2400}

# Engine sizing, all figures measured on this node in MiB against the card's
# 81,559 MiB. vLLM overshoots its own target by ~2,100 MiB of non-torch
# allocations (NCCL buffers, CUDA context), so 0.78 produced a 65,719 MiB awake
# footprint - only 1,600 MiB clear of the trainer's 14,234 MiB wake-time
# residual.
#
# 0.58 lands at ~49,400 MiB awake with ~18 GiB of KV per GPU (~386k tokens per
# replica, ~85 concurrent judge requests). Measured peak demand is 47 concurrent
# with 0 queued, so this is still over-provisioned; the memory it gives back is
# what keeps the actor parameters and optimizer resident.
export IF_LLM_VERIFIER_DTYPE=${IF_LLM_VERIFIER_DTYPE:-bfloat16}
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.78}
# Judge prompt = rubric + x (<=2048 tok) + y (<=2048 tok); 12288 is ample.
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-12288}
# ~60 GB of KV per replica is ~625k tokens, i.e. ~139 concurrent judge requests
# at ~4.5k tokens each; size the scheduler to actually use it.
export IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS=${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS:-16384}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-128}
# The judge decodes ~30 tokens per call, so CUDA graphs buy little and cost
# both capture time (x8 replicas) and the GPU memory the KV cache wants.
export IF_LLM_VERIFIER_ENFORCE_EAGER=${IF_LLM_VERIFIER_ENFORCE_EAGER:-true}
export IF_LLM_VERIFIER_TRUST_REMOTE_CODE=${IF_LLM_VERIFIER_TRUST_REMOTE_CODE:-false}
# vLLM's default TP all-reduce path rendezvouses through PyTorch symmetric
# memory, which needs CUDA multicast/VMM privileges this container does not
# grant: TP>1 dies in torch_symm_mem.rendezvous with "CUDA driver error: the
# operation cannot be performed in the present state". Falling back to NCCL
# costs nothing on an NVLink pair. Harmless at TP=1.
export IF_LLM_VERIFIER_USE_SYMM_MEM=${IF_LLM_VERIFIER_USE_SYMM_MEM:-0}

# Reward-side judging policy.
export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-7}
export IF_LLM_VERIFIER_ANCHOR_FALLBACK_ONLY=${IF_LLM_VERIFIER_ANCHOR_FALLBACK_ONLY:-true}
export IF_LLM_VERIFIER_TEMPERATURE=${IF_LLM_VERIFIER_TEMPERATURE:-0.0}
export IF_LLM_VERIFIER_TOP_P=${IF_LLM_VERIFIER_TOP_P:-1.0}
# The judge answers with `<think></think>` + `{"Score": N}` - about 17 tokens.
# At 1024 the handful of rows per step that make it ramble instead run to the
# cap, and because a lone request decodes at only ~26 tok/s that is ~39s per
# attempt, ~118s across the 3 attempts max_retries allows. Measured: the bulk of
# ~3,300 judge calls finishes in ~30s and then ONE straggler holds all 8 GPUs
# idle for ~110s, which is what makes timing_s/reward swing 84s <-> 189s on
# identical workload. 128 bounds that tail to ~5s per attempt and is still 7x
# the length a well-formed answer needs; extract_judge_score also has a regex
# fallback that recovers the score from truncated JSON.
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-128}
export IF_LLM_VERIFIER_OMIT_MAX_TOKENS=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS:-false}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}
# A whole fallback phase is issued at once, so a request can wait behind the
# rest of the batch; time out generously rather than silently dropping bonuses.
export IF_LLM_VERIFIER_TIMEOUT=${IF_LLM_VERIFIER_TIMEOUT:-900}
export IF_LLM_VERIFIER_MAX_RETRIES=${IF_LLM_VERIFIER_MAX_RETRIES:-2}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-64}

########################### checkpoints, wandb, Hub ###########################
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
PY_GIVEN_X_REWARD_TAG=${PY_GIVEN_X_REWARD_COEFF/./}
IF_LLM_VERIFIER_BONUS_TAG=${IF_LLM_VERIFIER_BONUS/./}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-llama31_tulu3_8b_dpo_grpo_nonthink_anchor_pyx${PY_GIVEN_X_REWARD_TAG}_llmverifier_qwen3_30ba3b_nonthink_fallback_bonus${IF_LLM_VERIFIER_BONUS_TAG}_threshold${IF_LLM_VERIFIER_THRESHOLD}_b${TRAIN_BATCH_SIZE}_c1_t1_2k}

CKPT_DIR=${CKPT_DIR:-${VERL_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${CKPT_DIR}"

if [[ -z "${WANDB_API_KEY:-}" && ! -s "${HOME}/.netrc" ]]; then
    echo "ERROR: wandb logging is required but no WANDB_API_KEY and no ~/.netrc were found." >&2
    echo "       export WANDB_API_KEY=... (or run 'wandb login') before starting this run." >&2
    exit 1
fi

HF_CHECKPOINT_PUSH=${HF_CHECKPOINT_PUSH:-true}
HF_CHECKPOINT_REPO_PRIVATE=${HF_CHECKPOINT_REPO_PRIVATE:-true}
HF_CHECKPOINT_POLL_SECONDS=${HF_CHECKPOINT_POLL_SECONDS:-120}
HF_CHECKPOINT_FINAL_SWEEP=${HF_CHECKPOINT_FINAL_SWEEP:-true}
HF_CHECKPOINT_PUSHER="${SCRIPT_DIR}/push_checkpoints_to_hf.py"
# EXPERIMENT_NAME is far longer than the Hub's 96-character repo-name limit, so
# the repo gets its own short, stable name.
export HF_CHECKPOINT_REPO_NAME=${HF_CHECKPOINT_REPO_NAME:-tulu3-8b-dpo-grpo-anchor-pyx${PY_GIVEN_X_REWARD_TAG}-q30ba3b-t${IF_LLM_VERIFIER_THRESHOLD}}
if [[ "${HF_CHECKPOINT_PUSH}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ -z "${HF_CHECKPOINT_REPO:-}" ]]; then
        # Default to <hub-user>/<short-name>; the token must be able to write it.
        HF_CHECKPOINT_REPO=$("${TULU3_PYTHON_BIN}" - <<'PY'
import os
import sys

from huggingface_hub import HfApi

try:
    user = HfApi(token=os.getenv("HF_TOKEN") or None).whoami()["name"]
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"cannot resolve the Hugging Face account for checkpoint upload: {type(exc).__name__}: {exc}\n"
        "Export a write-scoped HF_TOKEN, or set HF_CHECKPOINT_REPO=<owner>/<name>, "
        "or disable uploading with HF_CHECKPOINT_PUSH=false."
    )
print(f"{user}/{os.environ['HF_CHECKPOINT_REPO_NAME'][:96]}")
PY
        ) || {
            echo "ERROR: could not determine the Hugging Face checkpoint repo (see the message above)." >&2
            exit 1
        }
    fi
    export HF_CHECKPOINT_REPO
    echo "[hf-push] checkpoints -> https://huggingface.co/${HF_CHECKPOINT_REPO} (private=${HF_CHECKPOINT_REPO_PRIVATE})"
else
    HF_CHECKPOINT_REPO=""
    echo "[hf-push] disabled (HF_CHECKPOINT_PUSH=${HF_CHECKPOINT_PUSH})"
fi

########################### banner ###########################
cat >&2 <<BANNER
[run] experiment      : ${EXPERIMENT_NAME}
[run] policy          : ${MODEL_PATH} @ ${TULU3_MODEL_REVISION}
[run] verifier        : ${IF_LLM_VERIFIER_MODEL} (threshold=${IF_LLM_VERIFIER_THRESHOLD}, bonus=${IF_LLM_VERIFIER_BONUS}, thinking=${IF_LLM_VERIFIER_ENABLE_THINKING})
[run] anchor cache    : ${IF_REF_ANCHOR_CACHE_PATH}
[run] gpus            : train=${GPU_SET} verifier=${IF_LLM_VERIFIER_GPU_SET} (tp=${IF_LLM_VERIFIER_TP})
[run] batch           : ${TRAIN_BATCH_SIZE} x n=${ROLLOUT_N}, ${TOTAL_EPOCHS} epochs, save every ${SAVE_FREQ} steps
[run] checkpoints     : ${CKPT_DIR}
[run] wandb           : ${WANDB_ENTITY}/${PROJECT_NAME}
BANNER

########################### weight prefetch ###########################
# Eight vLLM servers starting at once would otherwise race on the same 61 GB
# download. Fetch both models exactly once, up front. Both repos ship complete
# safetensors, so the duplicated .bin/.pth copies are skipped.
echo "[setup] prefetching model weights into ${HF_HUB_CACHE}"
"${TULU3_PYTHON_BIN}" - <<'PY'
import os

from huggingface_hub import snapshot_download

IGNORE = ["*.pth", "*.bin", "*.bin.index.json", "original/*", "consolidated*"]

# The policy is fetched at `main`; tulu3_require_runtime has already asserted
# that `main` is TULU3_MODEL_REVISION, and verl itself resolves `main`.
targets = [(os.environ["MODEL_PATH"], None)]
verifier = os.environ["IF_LLM_VERIFIER_MODEL"]
if not os.path.isdir(verifier):
    # The judge servers are launched with --revision, so pin the same snapshot.
    targets.append((verifier, os.environ.get("IF_LLM_VERIFIER_REVISION") or None))

for repo_id, revision in targets:
    if os.path.isdir(repo_id):
        continue
    path = snapshot_download(repo_id, revision=revision, ignore_patterns=IGNORE)
    print(f"[setup] ready: {repo_id} -> {path}")
PY

########################### verifier servers ###########################
VERIFIER_PIDS=()
VERIFIER_LOG_FILES=()
VERIFIER_BASE_URL_ARRAY=()
HF_PUSH_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${HF_PUSH_PID}" ]]; then
        kill "${HF_PUSH_PID}" 2>/dev/null || true
        wait "${HF_PUSH_PID}" 2>/dev/null || true
        HF_PUSH_PID=""
    fi
    if ((${#VERIFIER_PIDS[@]})); then
        for pid in "${VERIFIER_PIDS[@]}"; do
            kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        done
        for pid in "${VERIFIER_PIDS[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
        VERIFIER_PIDS=()
    fi
    if [[ -n "${HF_CHECKPOINT_REPO:-}" && \
          "${HF_CHECKPOINT_FINAL_SWEEP}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "[hf-push] final sweep for ${CKPT_DIR}" >&2
        local -a sweep_args=(
            --ckpt-dir "${CKPT_DIR}"
            --repo-id "${HF_CHECKPOINT_REPO}"
            --run-name "${EXPERIMENT_NAME}"
            --base-model "${MODEL_PATH}"
            --once
        )
        if [[ "${HF_CHECKPOINT_REPO_PRIVATE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            sweep_args+=(--private)
        fi
        "${TULU3_PYTHON_BIN}" "${HF_CHECKPOINT_PUSHER}" "${sweep_args[@]}" \
            || echo "[hf-push] final sweep failed; re-run push_checkpoints_to_hf.py manually." >&2
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
        replica_log="${IF_LLM_VERIFIER_LOG_DIR}/qwen3_30ba3b_gpu${replica_gpu_csv//,/_}_${replica_port}.log"

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
    # Hand the GPUs back to the trainer before it profiles its own memory.
    for base_url in "${VERIFIER_BASE_URL_ARRAY[@]}"; do
        IF_LLM_VERIFIER_CONTROL_URL="${base_url}" \
            bash "${SCRIPT_DIR}/control_qwen3_30ba3b_verifier.sh" sleep "${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    done
fi
########################### end verifier servers ###########################

########################### checkpoint -> Hub watcher ###########################
if [[ "${HF_CHECKPOINT_PUSH}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    HF_PUSH_ARGS=(
        --ckpt-dir "${CKPT_DIR}"
        --repo-id "${HF_CHECKPOINT_REPO}"
        --run-name "${EXPERIMENT_NAME}"
        --base-model "${MODEL_PATH}"
        --poll-seconds "${HF_CHECKPOINT_POLL_SECONDS}"
    )
    if [[ "${HF_CHECKPOINT_REPO_PRIVATE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        HF_PUSH_ARGS+=(--private)
    fi
    mkdir -p "${VERL_DIR}/logs"
    "${TULU3_PYTHON_BIN}" "${HF_CHECKPOINT_PUSHER}" "${HF_PUSH_ARGS[@]}" \
        >"${VERL_DIR}/logs/hf_push_${EXPERIMENT_NAME}.log" 2>&1 &
    HF_PUSH_PID=$!
    echo "[hf-push] watcher pid=${HF_PUSH_PID} log=${VERL_DIR}/logs/hf_push_${EXPERIMENT_NAME}.log" >&2
fi

########################### actor training ###########################
REWARD_MANAGER_PATH="${VERL_DIR}/if_rlvr/if_llm_verifier_reward_manager.py"

OVERRIDES=(
    # --- Tulu 3 rollout / precision profile -------------------------------
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
    # --- host residency so the awake verifier fits on the same GPU --------
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
    # --- checkpoints: keep the resumable shards AND export an HF model ----
    trainer.default_local_dir="${CKPT_DIR}"
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]"
    "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]"
    # --- ray runtime env --------------------------------------------------
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=${HF_HOME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=${HF_HUB_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    # Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It looks
    # like an easy way to shrink the ref-policy pass's allocator residue, but
    # vLLM's sleep mode allocates through a CUDA memory pool and asserts
    # "Expandable segments are not compatible with memory pool", killing every
    # rollout engine at init. verl itself toggles it off around weight sync.
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_APPLY_ENABLE_THINKING_KWARG=\"${IF_APPLY_ENABLE_THINKING_KWARG}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_ALLOW_MISSING_THINK_FINAL_ANSWER=\"${IF_ALLOW_MISSING_THINK_FINAL_ANSWER}\""
    # --- reward: IFEval constraint + anchor interval + verifier fallback ---
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
    "+reward.reward_kwargs.if_llm_verifier_enable_thinking=${IF_LLM_VERIFIER_ENABLE_THINKING}"
    "+reward.reward_kwargs.if_llm_verifier_bonus=${IF_LLM_VERIFIER_BONUS}"
    "+reward.reward_kwargs.if_llm_verifier_threshold=${IF_LLM_VERIFIER_THRESHOLD}"
    "+reward.reward_kwargs.if_llm_verifier_anchor_fallback_only=${IF_LLM_VERIFIER_ANCHOR_FALLBACK_ONLY}"
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

set +e
bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" "${OVERRIDES[@]}" "$@"
status=$?
set -e
exit "${status}"
