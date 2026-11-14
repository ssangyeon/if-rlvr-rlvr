#!/usr/bin/env bash
# Constraint-only GRPO at KL coefficients 0.02 and 0.05, four policies, 4 epochs each, on one
# 8x H200 node run as TWO concurrent 4-GPU slots (GPUs 0-3 and 4-7).
#
#   #  job                       policy                          mode       wandb experiment_name
#   ---------------------------------------------------------------------------------------------------------------
#   1  tulu3_8b_kl0p02           allenai/Llama-3.1-Tulu-3-8B-DPO  nonthink   llama31_tulu3_8b_dpo_grpo_nonthink_constraint_only_b1024_c1_kl0p02
#   2  tulu3_8b_kl0p05           allenai/Llama-3.1-Tulu-3-8B-DPO  nonthink   llama31_tulu3_8b_dpo_grpo_nonthink_constraint_only_b1024_c1_kl0p05
#   3  qwen3_4b_nothink_kl0p02   Qwen/Qwen3-4B                    nonthink   qwen3_4b_grpo_nonthink_constraint_only_b1024_c1_kl0p02
#   4  qwen3_4b_nothink_kl0p05   Qwen/Qwen3-4B                    nonthink   qwen3_4b_grpo_nonthink_constraint_only_b1024_c1_kl0p05
#   5  qwen3_17b_think_kl0p02    Qwen/Qwen3-1.7B                  think      qwen3_17b_grpo_think_constraint_only_b1024_c1_kl0p02
#   6  qwen3_17b_think_kl0p05    Qwen/Qwen3-1.7B                  think      qwen3_17b_grpo_think_constraint_only_b1024_c1_kl0p05
#   7  qwen3_4b_think_kl0p02     Qwen/Qwen3-4B                    think      qwen3_4b_grpo_think_constraint_only_b1024_c1_kl0p02
#   8  qwen3_4b_think_kl0p05     Qwen/Qwen3-4B                    think      qwen3_4b_grpo_think_constraint_only_b1024_c1_kl0p05
#
# Each job's checkpoints (bf16 HF export of every save) are pushed as they land to the PUBLIC repo
#   <hf user>/{llama31-tulu3-8b-dpo,qwen3-4b,qwen3-1.7b-think,qwen3-4b-think}-if-rlvr-constraint-only-kl0p0{2,5}
# one folder per step (global_step_91/182/273/364), by push_checkpoints_to_hf.py.
#
# SCHEDULING. The two slots pull from ONE ordered queue, so phases run in the order above and the
# two KL arms of a policy start together. A slot that finishes early simply takes the next job
# instead of idling until its partner is done. Re-running this script after an interruption skips
# jobs with a .done marker and resumes the rest from their newest checkpoint (resume_mode=auto).
#
# RECIPES (unchanged from the leaf scripts; only kl_loss_coef differs from the originals):
#   tulu3_8b        llama31_tulu3_8b_dpo_constraint_only_nonreason.sh  (2048/2048, lr 5e-7, top_p 0.95, data.seed 42)
#   qwen3_4b_nothink qwen3_4b_constraint_only_nonreason.sh             (2048/2048, lr 5e-7)
#   qwen3_17b_think qwen3_17b_constraint_only_reasoning.sh             (2048/8192, lr 1e-6)
#   qwen3_4b_think  qwen3_17b_constraint_only_reasoning.sh with MODEL_PATH=Qwen/Qwen3-4B
#                   (= the qwen3_4b_grpo_think_constraint_only_b1024_c1 recipe, wandb roixvo8n)
#   All: train_batch 1024, ppo_mini_batch 1024, n=8, low_var_kl, SAVE_FREQ=91; TOTAL_EPOCHS=4 for
#   the non-reasoning runs and Qwen3-1.7B think, 3 for Qwen3-4B think.
#   The reasoning leaf defaults MAX_PROMPT_LENGTH=8192/TOTAL_EPOCHS=5; both are overridden to the
#   2048 / 4 that the recorded think baselines (wandb i4fetmrx, roixvo8n) actually ran with, so the
#   filtered train split and the 91 steps/epoch match them. Qwen runs pin data.seed=42 (the Tulu
#   leaf already does) so the two KL arms of a policy see the same batch order.
#
# THROUGHPUT knobs (vLLM memory/concurrency, micro-batch token caps, gradient checkpointing) live
# in kl_sweep_h200_tuning.sh, which is re-sourced at the start of every job.
#
# Usage:
#   setsid nohup bash if_rlvr/exps/bidirectional/run_constraint_only_kl_sweep_h200.sh \
#       > logs/kl_sweep_h200/orchestrator.log 2>&1 < /dev/null &
#   DRY_RUN=1 bash .../run_constraint_only_kl_sweep_h200.sh        # print resolved commands only
#   KLSWEEP_JOBS="qwen3_4b_think_kl0p02" bash .../run_...sh         # subset / custom order
# Stop:
#   bash if_rlvr/exps/bidirectional/stop_constraint_only_kl_sweep_h200.sh          # everything
#   SLOT=1 bash .../stop_constraint_only_kl_sweep_h200.sh                          # one slot's job

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
cd "${VERL_DIR}"

TUNING_FILE=${TUNING_FILE:-${SCRIPT_DIR}/kl_sweep_h200_tuning.sh}
STOP_SCRIPT=${SCRIPT_DIR}/stop_constraint_only_kl_sweep_h200.sh
PUSH_SCRIPT=${SCRIPT_DIR}/push_checkpoints_to_hf.py
LOG_DIR=${LOG_DIR:-${VERL_DIR}/logs/kl_sweep_h200}
STATE_DIR=${STATE_DIR:-${LOG_DIR}/state}
CKPT_ROOT=${CKPT_ROOT:-${VERL_DIR}/checkpoints/verl_if_rlvr}
DRY_RUN=${DRY_RUN:-0}
SLOTS=${SLOTS:-"0 1"}
GPUS_PER_SLOT=${GPUS_PER_SLOT:-4}
SLOT_START_STAGGER_S=${SLOT_START_STAGGER_S:-60}
KLSWEEP_JOBS=${KLSWEEP_JOBS:-"tulu3_8b_kl0p02 tulu3_8b_kl0p05 qwen3_4b_nothink_kl0p02 qwen3_4b_nothink_kl0p05 qwen3_17b_think_kl0p02 qwen3_17b_think_kl0p05 qwen3_4b_think_kl0p02 qwen3_4b_think_kl0p05"}

mkdir -p "${LOG_DIR}" "${STATE_DIR}" "${CKPT_ROOT}"

########################### runtime environment ###########################
PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    else
        PYTHON_BIN=$(command -v python3)
    fi
fi
# _llama31_tulu3_8b_common.sh otherwise tries to activate ${IFIF_ROOT}/.miniforge3/envs/verl and
# run ${IFIF_ROOT}/.venvs/verl-olmo3/bin/python, neither of which exists on this box.
export TULU3_CONDA_ENV_DIR=${TULU3_CONDA_ENV_DIR:-${CONDA_PREFIX:-$(dirname -- "$(dirname -- "${PYTHON_BIN}")")}}
export TULU3_PYTHON_BIN=${TULU3_PYTHON_BIN:-${PYTHON_BIN}}
# The ambient HF_HOME (/workspace/.cache/huggingface) does not exist here; keep weights and the
# dataset on the repo's own local disk.
export HF_HOME=${KLSWEEP_HF_HOME:-${VERL_DIR}/.cache/huggingface}
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
if [[ -z "${NLTK_DATA_DIR:-}" && -d "${HOME}/nltk_data/tokenizers/punkt_tab" ]]; then
    export NLTK_DATA_DIR="${HOME}/nltk_data"
fi
# W&B destination exactly as in the leaf scripts.
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
unset WANDB_RUN_ID WANDB_RESUME

log() { echo "[klsweep $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ "${DRY_RUN}" != "1" ]]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY is not set (trainer.logger includes wandb)}"
fi
HF_USER=${HF_USER:-$("${PYTHON_BIN}" -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])')}

########################### job definitions ###########################
# job_setup <job> -> JOB_LEAF JOB_EXP JOB_REPO JOB_BASE_MODEL JOB_ENV[] JOB_HYDRA[]
job_setup() {
    local job=$1
    # shellcheck source=kl_sweep_h200_tuning.sh
    source "${TUNING_FILE}"

    local family=${job%_kl*}
    local tag=${job##*_}
    if [[ ! "${tag}" =~ ^kl[0-9]+p[0-9]+$ ]]; then
        echo "bad job name ${job}: expected <family>_kl<int>p<frac>" >&2
        return 2
    fi
    local coef=${tag#kl}
    coef=${coef/p/.}

    local prefix max_model_len
    JOB_ENV=(
        "KL_LOSS_COEF=${coef}"
        "TOTAL_EPOCHS=4"
        "SAVE_FREQ=91"
        "TRAIN_BATCH_SIZE=1024"
        "PPO_MINI_BATCH_SIZE=1024"
        "ROLLOUT_N=8"
        "AGENT_NUM_WORKERS=64"
        "DATA_PROCESSOR_CPU_COUNT=32"
    )
    JOB_HYDRA=()
    case "${family}" in
        tulu3_8b)
            prefix=TULU
            JOB_LEAF=llama31_tulu3_8b_dpo_constraint_only_nonreason.sh
            JOB_BASE_MODEL=allenai/Llama-3.1-Tulu-3-8B-DPO
            JOB_EXP=llama31_tulu3_8b_dpo_grpo_nonthink_constraint_only_b1024_c1_${tag}
            JOB_REPO=${HF_USER}/llama31-tulu3-8b-dpo-if-rlvr-constraint-only-${tag}
            max_model_len=4096
            ;;
        qwen3_4b_nothink)
            prefix=Q4BNT
            JOB_LEAF=qwen3_4b_constraint_only_nonreason.sh
            JOB_BASE_MODEL=Qwen/Qwen3-4B
            JOB_EXP=qwen3_4b_grpo_nonthink_constraint_only_b1024_c1_${tag}
            JOB_REPO=${HF_USER}/qwen3-4b-if-rlvr-constraint-only-${tag}
            max_model_len=4096
            JOB_HYDRA+=("data.seed=42")
            ;;
        qwen3_17b_think)
            prefix=Q17BT
            JOB_LEAF=qwen3_17b_constraint_only_reasoning.sh
            JOB_BASE_MODEL=Qwen/Qwen3-1.7B
            JOB_EXP=qwen3_17b_grpo_think_constraint_only_b1024_c1_${tag}
            JOB_REPO=${HF_USER}/qwen3-1.7b-think-if-rlvr-constraint-only-${tag}
            # Qwen3-1.7B reasoning trains 4 epochs (364 steps; user request 2026-11-12, extended from
            # 3 at the step-273 checkpoint). env's last TOTAL_EPOCHS wins.
            JOB_ENV+=("MAX_PROMPT_LENGTH=2048" "MAX_RESPONSE_LENGTH=8192" "TOTAL_EPOCHS=4")
            max_model_len=10240
            JOB_HYDRA+=("data.seed=42")
            ;;
        qwen3_4b_think)
            prefix=Q4BT
            JOB_LEAF=qwen3_17b_constraint_only_reasoning.sh
            JOB_BASE_MODEL=Qwen/Qwen3-4B
            JOB_EXP=qwen3_4b_grpo_think_constraint_only_b1024_c1_${tag}
            JOB_REPO=${HF_USER}/qwen3-4b-think-if-rlvr-constraint-only-${tag}
            JOB_ENV+=(
                "MODEL_PATH=Qwen/Qwen3-4B"
                "IF_REF_VLLM_MODEL=Qwen/Qwen3-4B"
                "MAX_PROMPT_LENGTH=2048"
                "MAX_RESPONSE_LENGTH=8192"
                "TOTAL_EPOCHS=3"
            )
            max_model_len=10240
            JOB_HYDRA+=("data.seed=42")
            ;;
        *)
            echo "unknown job family: ${family} (job ${job})" >&2
            return 2
            ;;
    esac

    local v
    v=${prefix}_GPU_UTIL;       local util=${!v}
    v=${prefix}_MAX_NUM_SEQS;   local seqs=${!v}
    v=${prefix}_BATCHED_TOKENS; local btok=${!v}
    v=${prefix}_GRAD_CKPT;      local gc=${!v}
    v=${prefix}_PPO_TOKENS;     local ppo=${!v}
    v=${prefix}_LOGPROB_TOKENS; local lp=${!v}

    JOB_ENV+=(
        "EXPERIMENT_NAME=${JOB_EXP}"
        "ROLLOUT_GPU_MEM_UTIL=${util}"
        "PPO_MAX_TOKEN_LEN_PER_GPU=${ppo}"
        "LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${lp}"
        # Strictly on-policy (ppo_mini_batch == train_batch, ppo_epochs 1): skip the redundant
        # old_log_prob forward pass; ppo_loss uses log_prob.detach() (same gradient, ~15% faster).
        # See verl/trainer/ppo/ray_trainer.py. Set KLSWEEP_SKIP_OLD_LOGPROB=0 to disable.
        "VERL_SKIP_OLD_LOGPROB_ON_POLICY=${KLSWEEP_SKIP_OLD_LOGPROB:-1}"
    )
    # Passed LAST, so they win over the leaf/base values (Hydra: last duplicate override wins).
    # gpu_memory_utilization must be a Hydra override: the Tulu leaf hard-assigns
    # ROLLOUT_GPU_MEM_UTIL=0.72 after sourcing its common file, so the env var cannot reach it.
    JOB_HYDRA+=(
        "actor_rollout_ref.rollout.gpu_memory_utilization=${util}"
        "actor_rollout_ref.rollout.max_model_len=${max_model_len}"
        "actor_rollout_ref.rollout.max_num_seqs=${seqs}"
        "actor_rollout_ref.rollout.max_num_batched_tokens=${btok}"
        "actor_rollout_ref.model.enable_gradient_checkpointing=${gc}"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo}"
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${lp}"
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${lp}"
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]"
        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]"
        "trainer.max_actor_ckpt_to_keep=1"
        "trainer.default_local_dir=${CKPT_ROOT}/${JOB_EXP}"
    )
    v="KLSWEEP_EXTRA_HYDRA_${job}"
    if [[ -n "${!v:-}" ]]; then
        local -a extra
        read -r -a extra <<<"${!v}"
        JOB_HYDRA+=("${extra[@]}")
    fi
}

# slot_setup <slot> -> SLOT_ENV[] SLOT_GPUS
slot_setup() {
    local slot=$1
    local first=$((slot * GPUS_PER_SLOT))
    SLOT_GPUS=$(seq -s, "${first}" "$((first + GPUS_PER_SLOT - 1))")
    SLOT_ENV=(
        "GPU_SET=${SLOT_GPUS}"
        "NGPUS_PER_NODE=${GPUS_PER_SLOT}"
        "RUN_SLOT=${slot}"
        # Ray worker-group ports 22000-22100 / 24000-24100; vLLM 25000-25399 / 26000-26399. All
        # below the 32768 ephemeral range, so Ray's own random listeners never squat on them.
        "IF_RLVR_PORT_BASE=$((22000 + 2000 * slot))"
        "VLLM_MASTER_PORT_BASE=$((25000 + 1000 * slot))"
        "VLLM_PORT_STRIDE=100"
        "VLLM_RESERVED_PORT_COUNT=16"
        "RAY_TMPDIR=/tmp/ifrlvr_r${slot}"
        "KLSWEEP_SLOT=${slot}"
    )
}

########################### helpers ###########################
slot_gpu_busy_count() {
    local gpus=$1
    nvidia-smi --id="${gpus}" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l
}

wait_slot_idle() {
    local slot=$1 gpus=$2 busy=0
    for _ in $(seq 1 60); do
        busy=$(slot_gpu_busy_count "${gpus}")
        (( busy == 0 )) && return 0
        sleep 5
    done
    log "slot ${slot}: WARNING ${busy} GPU(s) of ${gpus} still hold >1GB"
    return 1
}

reap_slot() {
    local slot=$1
    QUIET=1 SLOT="${slot}" bash "${STOP_SCRIPT}" || true
}

claim_next_job() {
    local slot=$1
    (
        flock 9
        for j in ${KLSWEEP_JOBS}; do
            [[ -e "${STATE_DIR}/${j}.done" || -e "${STATE_DIR}/${j}.claimed" || -e "${STATE_DIR}/${j}.failed" ]] && continue
            echo "slot=${slot} orchestrator_pid=$$ at=$(date '+%F %T')" >"${STATE_DIR}/${j}.claimed"
            echo "${j}"
            break
        done
    ) 9>"${STATE_DIR}/queue.lock"
}

# verify_upload <ckpt_dir> <repo> -> 0 when every saved step is on the Hub
verify_upload() {
    "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi

ckpt, repo = Path(sys.argv[1]), sys.argv[2]
try:
    last = int((ckpt / "latest_checkpointed_iteration.txt").read_text().strip())
except (OSError, ValueError):
    print(f"[verify] no completed checkpoint under {ckpt}")
    sys.exit(1)
want = set(range(91, last + 1, 91)) | {last}
api = HfApi()
files = set(api.list_repo_files(repo))
have = set()
for step in sorted(want):
    d = f"global_step_{step}"
    if f"{d}/config.json" not in files:
        continue
    # every weight shard must be on the hub, not just the config
    if f"{d}/model.safetensors.index.json" in files:
        import json
        idx = json.loads(Path(api.hf_hub_download(repo, f"{d}/model.safetensors.index.json")).read_text())
        shards = set(idx["weight_map"].values())
        if not all(f"{d}/{s}" in files for s in shards):
            continue
    elif f"{d}/model.safetensors" not in files:
        continue
    have.add(step)
missing = sorted(want - have)
print(f"[verify] {repo}: last local step {last}, complete on hub {sorted(have)}, missing {missing}")
sys.exit(1 if missing else 0)
PY
}

run_job() {
    local slot=$1 job=$2
    job_setup "${job}"
    slot_setup "${slot}"
    local ckpt="${CKPT_ROOT}/${JOB_EXP}"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local train_log="${LOG_DIR}/${job}.train.${ts}.log"
    local push_log="${LOG_DIR}/${job}.push.log"

    log "slot ${slot} (GPUs ${SLOT_GPUS}): START ${job}"
    log "  leaf    : ${JOB_LEAF}"
    log "  wandb   : ${WANDB_ENTITY}/${PROJECT_NAME} ${JOB_EXP}"
    log "  hub     : https://huggingface.co/${JOB_REPO} (public)"
    log "  ckpt    : ${ckpt}"
    log "  log     : ${train_log}"
    log "  env     : ${SLOT_ENV[*]} ${JOB_ENV[*]}"
    log "  hydra   : ${JOB_HYDRA[*]}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0   # keep the .claimed marker so the dry-run queue advances
    fi

    mkdir -p "${ckpt}"
    KLSWEEP_SLOT="${slot}" "${PYTHON_BIN}" "${PUSH_SCRIPT}" \
        --ckpt-dir "${ckpt}" --repo-id "${JOB_REPO}" --run-name "${JOB_EXP}" \
        --base-model "${JOB_BASE_MODEL}" --poll-seconds 120 >>"${push_log}" 2>&1 &
    local push_pid=$!

    local started=${SECONDS} rc
    set +e
    env "${SLOT_ENV[@]}" "${JOB_ENV[@]}" "IF_RLVR_RUN_ID=klsweep_${job}" \
        bash "${SCRIPT_DIR}/${JOB_LEAF}" "${JOB_HYDRA[@]}" >"${train_log}" 2>&1
    rc=$?
    set -e
    local mins=$(( (SECONDS - started) / 60 ))
    log "slot ${slot}: ${job} trainer exited rc=${rc} after ${mins} min"

    kill "${push_pid}" 2>/dev/null || true
    wait "${push_pid}" 2>/dev/null || true
    KLSWEEP_SLOT="${slot}" "${PYTHON_BIN}" "${PUSH_SCRIPT}" \
        --ckpt-dir "${ckpt}" --repo-id "${JOB_REPO}" --run-name "${JOB_EXP}" \
        --base-model "${JOB_BASE_MODEL}" --once >>"${push_log}" 2>&1 || true

    local verified=1
    verify_upload "${ckpt}" "${JOB_REPO}" >>"${push_log}" 2>&1 && verified=0
    tail -n 1 "${push_log}" | sed 's/^/  /'

    reap_slot "${slot}"
    wait_slot_idle "${slot}" "${SLOT_GPUS}" || true

    rm -f "${STATE_DIR}/${job}.claimed"
    if (( rc == 0 && verified == 0 )); then
        echo "finished=$(date '+%F %T') minutes=${mins}" >"${STATE_DIR}/${job}.done"
        log "slot ${slot}: DONE ${job} -> https://huggingface.co/${JOB_REPO}"
        # Trainer finished every epoch (rc 0) and every saved step is complete on the Hub:
        # free the local checkpoint (~120 GB for Tulu). KLSWEEP_KEEP_LOCAL_CKPT=1 keeps it.
        if [[ "${KLSWEEP_KEEP_LOCAL_CKPT:-0}" != "1" && -n "${JOB_EXP}" && -d "${ckpt}" ]]; then
            rm -rf -- "${ckpt}"
            log "slot ${slot}: removed local checkpoints ${ckpt} (all steps verified on the Hub)"
        fi
        return 0
    fi
    echo "rc=${rc} upload_verified=$(( verified == 0 )) at=$(date '+%F %T') log=${train_log}" >"${STATE_DIR}/${job}.failed"
    log "slot ${slot}: FAILED ${job} (rc=${rc}, upload verified=$(( verified == 0 ))); see ${train_log} / ${push_log}"
    return 1
}

worker() {
    local slot=$1
    export KLSWEEP_SLOT=${slot}
    local job
    while true; do
        job=$(claim_next_job "${slot}")
        [[ -z "${job}" ]] && break
        slot_setup "${slot}"
        if [[ "${DRY_RUN}" != "1" ]]; then
            reap_slot "${slot}"
            if ! wait_slot_idle "${slot}" "${SLOT_GPUS}"; then
                rm -f "${STATE_DIR}/${job}.claimed"
                log "slot ${slot}: GPUs not free, not starting ${job}; slot stops"
                return 1
            fi
        fi
        if ! run_job "${slot}" "${job}"; then
            log "slot ${slot}: stopping after a failed job (the other slot keeps going)"
            return 1
        fi
    done
    log "slot ${slot}: queue empty, exiting"
}

########################### main ###########################
log "8x H200 constraint-only KL sweep; hub user ${HF_USER}; jobs: ${KLSWEEP_JOBS}"
# Claims from a dead orchestrator are stale; failed jobs are retried on a re-run.
# KLSWEEP_KEEP_CLAIMS=1: join a queue another orchestrator instance is still serving (e.g. to
# restart one slot) without stealing the jobs it has claimed.
if [[ "${KLSWEEP_KEEP_CLAIMS:-0}" != "1" ]]; then
    rm -f "${STATE_DIR}"/*.claimed "${STATE_DIR}"/*.failed
fi
for j in ${KLSWEEP_JOBS}; do
    job_setup "${j}"
    if [[ -e "${STATE_DIR}/${j}.done" ]]; then
        log "  [done] ${j}"
    else
        log "  [todo] ${j} -> ${JOB_EXP}"
    fi
done

avail_gb=$(df -BG --output=avail "${CKPT_ROOT}" | tail -1 | tr -dc '0-9')
log "free disk under ${CKPT_ROOT}: ${avail_gb} GB (peak need ~450 GB during the two Tulu saves)"

declare -a WORKER_PIDS=()
first=1
for slot in ${SLOTS}; do
    if (( first == 0 )) && [[ "${DRY_RUN}" != "1" ]]; then
        # Stagger engine/compile start-up so the two slots don't hit the HF cache, inductor cache
        # and CPU-heavy dataset filtering at the same moment.
        sleep "${SLOT_START_STAGGER_S}"
    fi
    first=0
    worker "${slot}" &
    WORKER_PIDS+=($!)
done

overall=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || overall=1
done
log "all slots finished (status ${overall})"
for j in ${KLSWEEP_JOBS}; do
    if [[ -e "${STATE_DIR}/${j}.done" ]]; then
        log "  DONE   ${j}: $(cat "${STATE_DIR}/${j}.done")"
    elif [[ -e "${STATE_DIR}/${j}.failed" ]]; then
        log "  FAILED ${j}: $(cat "${STATE_DIR}/${j}.failed")"
    else
        log "  NOTRUN ${j}"
    fi
done
exit "${overall}"
