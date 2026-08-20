#!/usr/bin/env bash
# Complete the fixed 20,480-row Qwen3-4B N=3 anchor panel.
#
# Default mode waits for GPUs 4-7 to become free, then runs run1, run2, and run3
# exactly once each. Use --preflight-only to validate everything without waiting
# for or touching a GPU.
set -Eeuo pipefail
umask 077

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd -- "${HERE}/../../.." && pwd)
IFIF_ROOT=$(cd -- "${VERL_DIR}/.." && pwd)
PYTHON_BIN=${VERL_ENV_PYTHON:-${IFIF_ROOT}/.miniforge3/envs/verl/bin/python}
PREPARE_TOOL="${HERE}/prepare_anchor_views.py"
RUNNER="${HERE}/generate_missing_run.sh"
RUNTIME_ROOT=${SUBSET20K_RUNTIME_ROOT:-${VERL_DIR}/.agent_runtime/subset20k}
RAY_TEMP_PARENT=${SUBSET20K_RAY_TEMP_PARENT:-${VERL_DIR}/.rt20k}
export SUBSET20K_RAY_TEMP_PARENT="${RAY_TEMP_PARENT}"
export HF_HOME=${HF_HOME:-/data/IFIF/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}
LOG_ROOT="${RUNTIME_ROOT}/logs"
STATUS_PATH="${RUNTIME_ROOT}/campaign_status.json"
PLAN_PATH="${RUNTIME_ROOT}/generation_plan.json"
LOCK_PATH="${RUNTIME_ROOT}/campaign.lock"
MODE=${1:-}

if [[ -n "${MODE}" && "${MODE}" != "--preflight-only" ]]; then
    echo "usage: $0 [--preflight-only]" >&2
    exit 2
fi

mkdir -p "${RUNTIME_ROOT}" "${LOG_ROOT}" "${RAY_TEMP_PARENT}"
exec 9>"${LOCK_PATH}"
flock -n 9 || { echo "[subset20k] another campaign process owns ${LOCK_PATH}" >&2; exit 4; }

write_status() {
    local state=$1
    local phase=${2:-none}
    local temporary="${STATUS_PATH}.tmp.$$"
    printf '{"state":"%s","phase":"%s","updated":"%s"}\n' \
        "${state}" "${phase}" "$(date --iso-8601=seconds)" \
        >"${temporary}"
    mv -f -- "${temporary}" "${STATUS_PATH}"
}

MONITOR_PID=
stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    MONITOR_PID=
}

fail_handler() {
    local rc=$?
    trap - ERR INT TERM
    set +e
    stop_monitor
    write_status failed "${CURRENT_PHASE:-preflight}"
    echo "[subset20k] campaign stopped in ${CURRENT_PHASE:-preflight} (rc=${rc}); no retry submitted." >&2
    exit "${rc}"
}
trap fail_handler ERR INT TERM
trap stop_monitor EXIT

CURRENT_PHASE=preflight
write_status preflight preflight
[[ -x "${PYTHON_BIN}" ]] || { echo "missing executable Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -x "${RUNNER}" ]] || { echo "runner is not executable: ${RUNNER}" >&2; exit 2; }
command -v nvidia-smi >/dev/null
command -v flock >/dev/null
bash -n "${RUNNER}"
bash -n "${BASH_SOURCE[0]}"
"${PYTHON_BIN}" -c \
    'from pathlib import Path; import sys; p=Path(sys.argv[1]); compile(p.read_text(encoding="utf-8"), str(p), "exec")' \
    "${PREPARE_TOOL}"

# Fail closed if someone edits away any throughput, sampling, or no-retry invariant.
"${PYTHON_BIN}" - "${RUNNER}" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = [
    "DEVICE=gpu",
    "INFER_BACKEND=vllm",
    "GPU_SET=4,5,6,7",
    "NGPUS_PER_NODE=4",
    "ROLLOUT_TP=1",
    "RAY_TEMP_PARENT=",
    "ROLLOUT_N=1",
    "IF_DATASET_HF=allenai/IF_multi_constraints_upto5",
    "IF_PPL_PREFIX_MODE=standard",
    "IF_REF_POLICY_ANCHOR_PPL=true",
    "ENABLE_THINKING=true",
    "IF_REQUIRE_THINK_END_FOR_REWARD=true",
    "IF_ALLOW_MISSING_THINK_FINAL_ANSWER=false",
    "MAX_PROMPT_LENGTH=2048",
    "MAX_RESPONSE_LENGTH=32768",
    "ROLLOUT_TEMPERATURE=1.0",
    "ROLLOUT_TOP_P=0.95",
    "ROLLOUT_TOP_K=20",
    "PRESENCE_PENALTY=0.0",
    "IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES=0",
    "IF_MAX_RETRIES=1",
    "IF_REF_ANCHOR_CACHE_METADATA_STRICT=true",
    "max_model_len=40960",
    "max_num_seqs=128",
    "max_num_batched_tokens=65536",
    "gpu_memory_utilization=0.55",
    "PRECOMPUTE_BATCH_ROWS=",
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
    "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
]
missing = [value for value in required if value not in text]
forbidden = ["retrying in", "IF_ANCHOR_HYGIENE_PASSES", "while true", "/tmp/subset20k_"]
present = [value for value in forbidden if value in text]
if missing or present:
    raise SystemExit(f"runner invariant failure: missing={missing}, forbidden={present}")
print("[subset20k] runner invariants: PASS")
PY

"${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
    validate-plan >"${PLAN_PATH}"
"${PYTHON_BIN}" - "${PLAN_PATH}" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert plan["selected_inputs"] == 20_480
assert plan["unique_inputs_requiring_generation"] == 3_034
assert plan["anchor_draws_to_generate"] == 4_789
assert plan["full_generation_completions"] == 9_578
assert plan["vllm_prompt_logprob_scoring_calls"] == 9_578
assert plan["execution"]["physical_gpus"] == [4, 5, 6, 7]
assert plan["execution"]["vllm_replicas"] == 4
assert plan["execution"]["tensor_model_parallel_size"] == 1
assert plan["execution"]["model_level_empty_response_retries"] == 0
assert plan["execution"]["launcher_retries"] == 0
assert plan["execution"]["campaign_retries"] == 0
expected_workers = {"run1": (72, 37, 0), "run2": (106, 10, 1), "run3": (82, 13, 0)}
for run, (workers, rows_per_worker, padding) in expected_workers.items():
    assert plan["runs"][run]["agent_workers"] == workers
    assert plan["runs"][run]["rows_per_agent_worker"] == rows_per_worker
    assert plan["runs"][run]["final_padding_requests"] == padding
    assert plan["runs"][run]["submitted_anchor_draws"] == workers * rows_per_worker
    assert plan["runs"][run]["missing_anchor_draws"] + padding == workers * rows_per_worker
print("[subset20k] exact workload accounting: PASS")
PY
export SUBSET20K_PLAN_VALIDATED=1

# The model and dataset must already be local. Preflight never falls back to a download.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${PYTHON_BIN}" - <<'PY'
from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
limit = int(getattr(config, "max_position_embeddings", 0))
assert limit >= 40_960, limit
assert tokenizer.chat_template, "Qwen3 tokenizer has no chat template"
print(f"[subset20k] local Qwen3-4B context={limit}: PASS")
PY

CPU_COUNT=$(nproc)
(( CPU_COUNT >= 176 )) || {
    echo "need at least 176 visible CPU cores for the configured feeder pool; found ${CPU_COUNT}" >&2
    exit 2
}

GPU_TABLE=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader,nounits)
for gpu in 4 5 6 7; do
    row=$(
        awk -F, -v wanted="${gpu}" \
            '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); if ($1 == wanted) print}' \
            <<<"${GPU_TABLE}")
    [[ -n "${row}" ]] || { echo "physical GPU ${gpu} is missing" >&2; exit 2; }
    total=$(awk -F, '{gsub(/[[:space:]]/, "", $3); print int($3)}' <<<"${row}")
    (( total >= 180000 )) || {
        echo "GPU ${gpu} has only ${total} MiB; expected a 183 GB-class device" >&2
        exit 2
    }
done

AVAILABLE_KIB=$(df -Pk "${RUNTIME_ROOT}" | awk 'NR==2 {print $4}')
MIN_FREE_KIB=104857600
(( AVAILABLE_KIB >= MIN_FREE_KIB )) || {
    echo "less than 100 GiB free at ${RUNTIME_ROOT}; refusing to begin" >&2
    exit 2
}
RAY_AVAILABLE_KIB=$(df -Pk "${RAY_TEMP_PARENT}" | awk 'NR==2 {print $4}')
(( RAY_AVAILABLE_KIB >= MIN_FREE_KIB )) || {
    echo "less than 100 GiB free at Ray temp parent ${RAY_TEMP_PARENT}; refusing to begin" >&2
    exit 2
}

NEEDS_GPU=false
for run in run1 run2 run3; do
    output="${RUNTIME_ROOT}/generated_runs/${run}.missing.cache.json"
    if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
        validate-generated-run --run "${run}" >/dev/null 2>&1; then
        echo "[subset20k] ${run}: existing output is complete and valid"
    elif "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
        audit-generated-run --run "${run}" >/dev/null 2>&1; then
        echo "[subset20k] ${run}: every one-shot attempt is already recorded; no retry"
    elif [[ -e "${output}" ]]; then
        echo "[subset20k] ${run}: structurally invalid/partial output exists: ${output}" >&2
        exit 3
    else
        echo "[subset20k] ${run}: ready for one fresh attempt"
        NEEDS_GPU=true
    fi
done

echo "[subset20k] preflight PASS: 4,789 anchor draws = 9,578 full completions."
echo "[subset20k] rollout topology: four full TP=1 replicas on physical GPUs 4,5,6,7."
if [[ "${MODE}" == "--preflight-only" ]]; then
    write_status preflight_passed none
    echo "[subset20k] preflight-only mode: no GPU process was started."
    exit 0
fi

if [[ "${NEEDS_GPU}" == true ]]; then
CURRENT_PHASE=wait_for_gpus
write_status waiting_for_gpus none
while :; do
    GPU_TABLE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    busy=$(awk -F, '
        {
            gsub(/[[:space:]]/, "", $1)
            gsub(/[[:space:]]/, "", $2)
            if (($1 + 0) >= 4 && ($1 + 0) <= 7 && ($2 + 0) > 4096) {
                printf "gpu%s=%sMiB ", $1, $2
            }
        }
    ' <<<"${GPU_TABLE}")
    if [[ -z "${busy}" ]]; then
        break
    fi
    echo "[subset20k] waiting for GPUs 4-7 to be free: ${busy}"
    sleep 30
done
echo "[subset20k] GPUs 4-7 are free; starting the first one-shot phase immediately."
fi

start_monitor() {
    local run=$1
    local monitor_log="${LOG_ROOT}/${run}.gpu.csv"
    (
        while :; do
            timestamp=$(date --iso-8601=seconds)
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits |
                awk -F, -v ts="${timestamp}" '
                    {
                        gsub(/[[:space:]]/, "", $1)
                        if ($1 >= 4 && $1 <= 7) print ts "," $0
                    }
                ' >>"${monitor_log}" || true
            for ((tick = 0; tick < 60; tick++)); do sleep 1; done
        done
    ) &
    MONITOR_PID=$!
}

ALL_STRICT=true
for run in run1 run2 run3; do
    CURRENT_PHASE=${run}
    write_status running "${run}"
    log="${LOG_ROOT}/${run}.generation.log"
    echo "[subset20k] ===== ${run} =====" | tee -a "${log}"
    start_monitor "${run}"
    "${RUNNER}" "${run}" 2>&1 | tee -a "${log}"
    stop_monitor
    "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
        audit-generated-run --run "${run}"
    if "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
        validate-generated-run --run "${run}" >/dev/null 2>&1; then
        "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" \
            finalize-run --run "${run}" --input-mode run
        echo "[subset20k] ${run}: finalized 20,480-row cache." | tee -a "${log}"
    else
        ALL_STRICT=false
        echo "[subset20k] ${run}: exact attempts complete, but cache is not train-ready; no retry." | tee -a "${log}"
    fi
done

CURRENT_PHASE=final_validation
write_status validating final
if [[ "${ALL_STRICT}" == true ]]; then
    "${PYTHON_BIN}" "${PREPARE_TOOL}" --runtime-root "${RUNTIME_ROOT}" validate-complete
    write_status complete none
    CURRENT_PHASE=complete
    echo "[subset20k] COMPLETE: all three 20,480-row anchor caches passed strict validation."
    exit 0
fi
write_status generation_complete_incomplete none
CURRENT_PHASE=generation_complete_incomplete
echo "[subset20k] all 4,789 one-shot draws were attempted, but at least one row is not train-ready." >&2
echo "[subset20k] no retry was submitted; inspect the per-run audit reports in the logs." >&2
exit 5
