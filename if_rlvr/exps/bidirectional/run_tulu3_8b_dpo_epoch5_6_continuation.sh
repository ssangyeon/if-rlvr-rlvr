#!/usr/bin/env bash
# Continues 4 already-4-epoch-trained Tulu3-8B-DPO IF-RLVR checkpoints for 2
# more epochs each (-> 6 epochs total), sequentially, on the full 8xH100 node,
# uploading every newly-completed epoch to the corresponding PUBLIC just1nseo
# HF repo. Runs, in order:
#
#   1. gpt-oss-120b judge (t5)  : just1nseo/llama31-tulu3-8b-dpo-if-rlvr-judge-only
#   2. constraint-only          : just1nseo/llama31-tulu3-8b-dpo-if-rlvr-constraint-only
#   3. anchor (pyx=0.1)         : sangyon/llama31_..._anchor_...  -> just1nseo/llama31-tulu3-8b-dpo-if-rlvr-anchor-only (new repo)
#   4. Qwen3-30B-A3B judge (t7) : just1nseo/tulu3-8b-dpo-grpo-q30ba3b-t7
#
# Why a fresh optimizer rather than a real verl `resume_mode=resume_path`: no
# local FSDP checkpoint (model/optimizer/extra shards) survives on this box for
# any of the 4 runs - only the bf16 `hf_model` export made it to the Hub. So
# each run here downloads global_step_364 (confirmed via HfApi as the final,
# 4-epoch checkpoint in all 4 source repos) and warm-starts a brand-new trainer
# instance from those weights, TOTAL_EPOCHS=2, constant LR (5e-7) so a reset
# optimizer/schedule is a non-issue. Its own step counter restarts at 0, so
# push_checkpoints_to_hf.py's --step-offset 364 is what keeps the new
# global_step_91/182-ish uploads from colliding with (overwriting) the
# already-uploaded global_step_91/182/273/364 from epochs 1-4.
#
# GPU plan: one run at a time uses the WHOLE node. The gpt-oss-120b judge does
# NOT need tensor-parallel sharding - its MXFP4 weights are 65.2 GB (confirmed
# via HfApi), well under one 80 GB H100, exactly like the two other
# gpt-oss-120b precedents already in this repo - so it runs as 8 TP=1
# replicas across all 8 GPUs, sharing them with the (offloaded) trainer via
# the same sleep/wake pattern as the Qwen3-30B-A3B judge script.
#
# Ref-model pinning (ALL 4 runs, not anchor-only): continuing training changes
# `actor_rollout_ref.model.path` to the epoch-4 checkpoint, and verl's `ref`
# submodule always defaults to mirroring actor/rollout's own weights unless
# told otherwise. Every one of these 4 leaf scripts sets use_kl_loss=True, so
# every one of them builds and uses a `ref` submodule for the KL penalty, not
# just the anchor run for its PPL reward - and epochs 1-4 could only ever have
# been regularized against the original base model (there was no earlier
# checkpoint to drift from), so epochs 5-6 must keep the same ref target for
# KL semantics to stay continuous across the epoch-4/5 seam and comparable
# across all 4 variants. IF_REF_POLICY_MODEL_PATH_OVERRIDE (opt-in, see
# verl/workers/engine_workers.py, no effect on any script that doesn't set it)
# pins ref back to allenai/Llama-3.1-Tulu-3-8B-DPO and is therefore passed to
# all 4 run_one calls below.
#
# This was originally wired for run 3 (anchor) only; run 1 (gpt-oss-120b
# judge) was actually launched and trained to step 101/182 without it on
# 2026-08-17 before the gap was caught, and was killed and restarted from
# scratch on 2026-08-18 specifically to fix this from step 0 - see the
# discarded checkpoint under
# checkpoints/verl_if_rlvr/tulu3_gptoss120b_judge_t5_epoch5to6.DISCARDED_wrongref_step101_2026-08-18/.
#
# Anchor run ALSO separately needs IF_REF_ANCHOR_CACHE_METADATA_STRICT=false:
# its precomputed teacher cache recorded the original base model's path as
# metadata, and the strict equality check would otherwise reject the whole
# (otherwise byte-identical) cache purely because `actor_rollout_ref.model.path`
# itself (as opposed to the ref override) now points at the epoch-4 checkpoint.
#
# Usage: nohup bash run_tulu3_8b_dpo_epoch5_6_continuation.sh >logs/epoch5_6_continuation/orchestrator.log 2>&1 &

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
cd "${VERL_DIR}"

# --- defensive interpreter resolution -----------------------------------
# _llama31_tulu3_8b_common.sh's own conda-activation fallback assumes a
# ${IFIF_ROOT}/.miniforge3 layout that does not exist on this box; pre-set
# TULU3_CONDA_ENV_DIR/TULU3_PYTHON_BIN to the environment that is ALREADY
# active (matching the "already active" shortcut some sibling scripts have
# built in) so every leaf script - whether or not it has that shortcut itself
# - skips the broken hardcoded path instead of failing at the first `source`.
if [[ -z "${TULU3_PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] && "${CONDA_PREFIX}/bin/python" -c 'import vllm' >/dev/null 2>&1; then
        export TULU3_CONDA_ENV_DIR="${CONDA_PREFIX}"
        export TULU3_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        echo "ERROR: no active conda env with vLLM importable (CONDA_PREFIX=${CONDA_PREFIX:-<unset>})." >&2
        exit 1
    fi
fi
PYTHON_BIN="${TULU3_PYTHON_BIN}"
echo "[orchestrator] using python: ${PYTHON_BIN}"

# --- HF cache placement --------------------------------------------------
# The inherited HF_HOME (/workspace/.cache/huggingface, from the shell
# profile) lives on a network-mounted, per-tenant-quota-limited volume
# (mfs#ca-mtl-1.runpod.net) whose `df` free-space figure is the CLUSTER-WIDE
# total, not this pod's actual quota - so the leaf scripts' own free-space
# safety check (>=120 GB) passes even though the real quota was already
# nearly exhausted (hit "Disk quota exceeded" 44 GB into the gpt-oss-120b
# download on the first real attempt). Force it onto the root overlay
# instead, which is genuine local disk (957 GB free, not quota-limited).
export HF_HOME="${VERL_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
echo "[orchestrator] HF_HOME=${HF_HOME} (forced off the quota-limited /workspace mount)"

ORIG_BASE_MODEL="allenai/Llama-3.1-Tulu-3-8B-DPO"
SOURCE_STEP=364
LOG_DIR="${VERL_DIR}/logs/epoch5_6_continuation"
CACHE_ROOT="${VERL_DIR}/.cache/epoch5_6_continuation"
mkdir -p "${LOG_DIR}" "${CACHE_ROOT}"

log() { echo "[orchestrator $(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

wait_for_gpus_idle() {
    local tries=0
    local max_tries=30
    while (( tries < max_tries )); do
        local used
        used=$( (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true) | awk '{sum+=$1} END {print sum+0}')
        if [[ -n "${used}" && "${used}" -lt 2000 ]]; then
            log "GPUs idle (total used: ${used} MiB)."
            return 0
        fi
        tries=$((tries + 1))
        log "waiting for GPUs to free up (used=${used:-unknown} MiB, attempt ${tries}/${max_tries})..."
        sleep 10
    done
    log "WARNING: GPUs did not fully clear after ${max_tries} attempts; proceeding anyway. Check for zombie processes if the next run misbehaves."
}

# run_one <name> <source_repo> <target_repo> <leaf_script> [extra hydra-override args...]
run_one() {
    local name=$1 source_repo=$2 target_repo=$3 leaf_script=$4
    shift 4
    local extra_args=("$@")

    log "=== [${name}] starting: ${source_repo}:global_step_${SOURCE_STEP} -> ${target_repo} (offset ${SOURCE_STEP}) ==="

    local local_ckpt_dir="${CACHE_ROOT}/${name}_src_step${SOURCE_STEP}"
    local model_path
    model_path=$("${PYTHON_BIN}" "${SCRIPT_DIR}/download_hf_checkpoint_step.py" \
        --repo-id "${source_repo}" --step "${SOURCE_STEP}" --local-dir "${local_ckpt_dir}")
    log "[${name}] MODEL_PATH=${model_path}"

    export MODEL_PATH="${model_path}"
    export TULU3_VERIFY_REMOTE_REVISIONS=false
    export TOTAL_EPOCHS=2
    export EXPERIMENT_NAME="${name}_epoch5to6"
    export CKPT_DIR="${VERL_DIR}/checkpoints/verl_if_rlvr/${EXPERIMENT_NAME}"
    mkdir -p "${CKPT_DIR}"

    local push_log="${LOG_DIR}/push_${name}.log"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/push_checkpoints_to_hf.py" \
        --ckpt-dir "${CKPT_DIR}" --repo-id "${target_repo}" --run-name "${EXPERIMENT_NAME}" \
        --base-model "${ORIG_BASE_MODEL}" --step-offset "${SOURCE_STEP}" --poll-seconds 120 \
        >"${push_log}" 2>&1 &
    local push_pid=$!
    log "[${name}] push watcher pid=${push_pid} log=${push_log}"

    local train_log="${LOG_DIR}/train_${name}.log"
    log "[${name}] training -> ${train_log}"
    set +e
    bash "${SCRIPT_DIR}/${leaf_script}" "${extra_args[@]}" >"${train_log}" 2>&1
    local status=$?
    set -e

    kill "${push_pid}" 2>/dev/null || true
    wait "${push_pid}" 2>/dev/null || true
    log "[${name}] final push sweep"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/push_checkpoints_to_hf.py" \
        --ckpt-dir "${CKPT_DIR}" --repo-id "${target_repo}" --run-name "${EXPERIMENT_NAME}" \
        --base-model "${ORIG_BASE_MODEL}" --step-offset "${SOURCE_STEP}" --once \
        >>"${push_log}" 2>&1 || log "[${name}] WARNING: final push sweep failed; re-run push_checkpoints_to_hf.py manually against ${CKPT_DIR}"

    if [[ ${status} -ne 0 ]]; then
        log "!!! [${name}] FAILED (exit ${status}). Stopping the sequence - see ${train_log}"
        exit "${status}"
    fi
    log "=== [${name}] done OK -> https://huggingface.co/${target_repo} ==="
    wait_for_gpus_idle
}

# START_FROM_RUN lets a relaunch skip already-completed runs (e.g. run 1
# succeeded, run 2 failed and needs a config fix - relaunching the whole
# script would otherwise redo run 1's already-uploaded 34h of work). Default
# 1 = original from-scratch behavior, unchanged.
export START_FROM_RUN=${START_FROM_RUN:-1}

log "starting 4-run epoch 5-6 continuation sequence (START_FROM_RUN=${START_FROM_RUN})"

# --- 1. gpt-oss-120b judge, threshold 5 ---------------------------------
if (( START_FROM_RUN <= 1 )); then
# Same ref-model-pinning reasoning as runs 2/4 (use_kl_loss=True builds a
# `ref` submodule here too). This run was actually launched once already
# without this override (2026-08-17), reached step 101/182 with ref silently
# mirroring the epoch-4 checkpoint instead of the original base model, and
# was killed and restarted from scratch on 2026-08-18 specifically to apply
# this fix from step 0 - see the discarded checkpoint at
# checkpoints/verl_if_rlvr/tulu3_gptoss120b_judge_t5_epoch5to6.DISCARDED_wrongref_step101_2026-08-18/
GPU_SET=0,1,2,3,4,5,6,7 NGPUS_PER_NODE=8 RUN_SLOT=0 \
IF_REF_POLICY_MODEL_PATH_OVERRIDE="${ORIG_BASE_MODEL}" \
run_one \
    "tulu3_gptoss120b_judge_t5" \
    "just1nseo/llama31-tulu3-8b-dpo-if-rlvr-judge-only" \
    "just1nseo/llama31-tulu3-8b-dpo-if-rlvr-judge-only" \
    "llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_POLICY_MODEL_PATH_OVERRIDE=${ORIG_BASE_MODEL}"
fi

# --- 2. constraint-only --------------------------------------------------
if (( START_FROM_RUN <= 2 )); then
# use_kl_loss=True here too (every variant's actor config sets it), so this
# run builds a `ref` submodule exactly like the anchor run does - it just
# doesn't consume it for a PPL reward, only the KL penalty. Same override,
# same reason: without it, ref silently mirrors actor/rollout's own weights
# (the epoch-4 checkpoint this run warm-starts from), not the original base
# model epochs 1-4 were regularized against.
GPU_SET=0,1,2,3,4,5,6,7 NGPUS_PER_NODE=8 RUN_SLOT=0 \
IF_REF_POLICY_MODEL_PATH_OVERRIDE="${ORIG_BASE_MODEL}" \
run_one \
    "tulu3_constraint_only" \
    "just1nseo/llama31-tulu3-8b-dpo-if-rlvr-constraint-only" \
    "just1nseo/llama31-tulu3-8b-dpo-if-rlvr-constraint-only" \
    "llama31_tulu3_8b_dpo_constraint_only_nonreason.sh" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_POLICY_MODEL_PATH_OVERRIDE=${ORIG_BASE_MODEL}"
fi

# --- 3. anchor (pyx=0.1) -------------------------------------------------
if (( START_FROM_RUN <= 3 )); then
# Original run used only GPU_SET=4,5,6,7 (RUN_SLOT=1, meant to share the node
# with a concurrent job); nothing else runs concurrently here, so use all 8.
# save_contents/default_local_dir are passed as extra Hydra overrides (not env
# vars) because this leaf script never sets them itself - the original repo's
# actual invocation must have passed the hf_model override the same way, since
# the Hydra default (['model','optimizer','extra'], no hf_model) could not have
# produced the HF-format global_step_364 that is already on the Hub.
GPU_SET=0,1,2,3,4,5,6,7 NGPUS_PER_NODE=8 RUN_SLOT=0 \
IF_REF_POLICY_MODEL_PATH_OVERRIDE="${ORIG_BASE_MODEL}" \
IF_REF_ANCHOR_CACHE_METADATA_STRICT=false \
TULU3_VALIDATE_ANCHOR_BEFORE_TRAIN=false \
run_one \
    "tulu3_anchor_pyx01" \
    "sangyon/llama31_tulu3_8b_dpo_grpo_nonthink_anchor_pyx01_b1024_c1_t1_2k" \
    "just1nseo/llama31-tulu3-8b-dpo-if-rlvr-anchor-only" \
    "llama31_tulu3_8b_dpo_anchor_grpo_nonreason.sh" \
    "trainer.default_local_dir=${VERL_DIR}/checkpoints/verl_if_rlvr/tulu3_anchor_pyx01_epoch5to6" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]" \
    "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_POLICY_MODEL_PATH_OVERRIDE=${ORIG_BASE_MODEL}"
fi

# --- 4. Qwen3-30B-A3B judge, threshold 7 --------------------------------
if (( START_FROM_RUN <= 4 )); then
# HF_CHECKPOINT_PUSH=false disables this script's OWN embedded pusher so it
# doesn't race the external, step-offset-aware watcher run_one already starts.
# Same ref-model-pinning reasoning as run 2 (use_kl_loss=True builds a `ref`
# submodule here too; without the override it silently mirrors the epoch-4
# checkpoint instead of the original base model).
GPU_SET=0,1,2,3,4,5,6,7 NGPUS_PER_NODE=8 RUN_SLOT=0 \
HF_CHECKPOINT_PUSH=false \
IF_REF_POLICY_MODEL_PATH_OVERRIDE="${ORIG_BASE_MODEL}" \
run_one \
    "tulu3_qwen30ba3b_judge_t7" \
    "just1nseo/tulu3-8b-dpo-grpo-q30ba3b-t7" \
    "just1nseo/tulu3-8b-dpo-grpo-q30ba3b-t7" \
    "llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_POLICY_MODEL_PATH_OVERRIDE=${ORIG_BASE_MODEL}"
fi

log "ALL 4 CONTINUATION RUNS COMPLETE"
