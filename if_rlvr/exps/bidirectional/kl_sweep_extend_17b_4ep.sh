#!/usr/bin/env bash
# One-shot helper (user request 2026-11-12): train qwen3_17b_think_kl0p02 for 4 epochs (364 steps)
# instead of 3. The running slot-0 orchestrator would, on the clean rc=0 exit at step 273, mark the
# job .done and delete its local checkpoints. So: detach (kill) that orchestrator now, let the
# trainer finish step 273 on its own, wait for global_step_273 to reach the Hub, then relaunch the
# slot with the current orchestrator (TOTAL_EPOCHS=4 for qwen3_17b_think), which resumes from 273.
# LR is constant (1e-6, no warmup), so this equals a native 4-epoch run.
#
#   setsid nohup bash kl_sweep_extend_17b_4ep.sh <orch_pids...> -- <leaf_pid> <push_pid> >>log 2>&1 &
set -u
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LOG_DIR=${VERL_DIR}/logs/kl_sweep_h200
STATE_DIR=${LOG_DIR}/state
JOB=qwen3_17b_think_kl0p02
CKPT=${VERL_DIR}/checkpoints/verl_if_rlvr/qwen3_17b_grpo_think_constraint_only_b1024_c1_kl0p02
PUSH_LOG=${LOG_DIR}/${JOB}.push.log
export KLSWEEP_JOBS="tulu3_8b_kl0p02 tulu3_8b_kl0p05 qwen3_4b_nothink_kl0p02 qwen3_4b_nothink_kl0p05 qwen3_17b_think_kl0p02 qwen3_4b_think_kl0p02 qwen3_4b_think_kl0p05"
log() { echo "[extend-17b-4ep $(date '+%F %T')] $*"; }

ORCH=()
while [[ $# -gt 0 && "$1" != "--" ]]; do ORCH+=("$1"); shift; done
shift
LEAF=$1 PUSH=$2

# sanity: the pids must still be slot 0's orchestrator / trainer leaf / push watcher
for p in "${ORCH[@]}"; do
    tr '\0' '\n' </proc/"${p}"/environ 2>/dev/null | grep -qx "SLOTS=0" || { log "pid ${p} is not a slot-0 orchestrator; abort"; exit 1; }
done
tr '\0' '\n' </proc/"${LEAF}"/environ 2>/dev/null | grep -qx "KLSWEEP_SLOT=0" || { log "leaf ${LEAF} not slot 0; abort"; exit 1; }
tr '\0' '\n' </proc/"${LEAF}"/environ 2>/dev/null | grep -qx "IF_RLVR_RUN_ID=klsweep_${JOB}" || { log "leaf ${LEAF} is not ${JOB}; abort"; exit 1; }
grep -q "qwen3_17b_grpo_think_constraint_only_b1024_c1_kl0p02" /proc/"${PUSH}"/cmdline || { log "push ${PUSH} mismatch; abort"; exit 1; }

log "detaching slot-0 orchestrator ${ORCH[*]} (trainer ${LEAF} and push watcher ${PUSH} keep running)"
kill -9 "${ORCH[@]}"

while kill -0 "${LEAF}" 2>/dev/null; do sleep 60; done
last=$(cat "${CKPT}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "?")
log "trainer exited; latest checkpoint global_step_${last}"

for _ in $(seq 1 80); do
    grep -q "uploaded global_step_${last} " "${PUSH_LOG}" 2>/dev/null && break
    sleep 30
done
grep -q "uploaded global_step_${last} " "${PUSH_LOG}" && log "global_step_${last} is on the Hub" \
    || log "WARNING global_step_${last} upload not seen yet; the relaunched push watcher will retry it"

log "pausing the watchdog, stopping slot 0"
pkill -STOP -f kl_sweep_h200_watchdog.py || true
trap 'pkill -CONT -f kl_sweep_h200_watchdog.py || true; log "watchdog resumed"' EXIT
kill "${PUSH}" 2>/dev/null || true
SLOT=0 QUIET=1 bash "${SCRIPT_DIR}/stop_constraint_only_kl_sweep_h200.sh" || true
rm -f "${STATE_DIR}/${JOB}.claimed" "${STATE_DIR}/${JOB}.failed" "${STATE_DIR}/${JOB}.done"

log "relaunching slot 0 (resumes ${JOB} from global_step_${last} with TOTAL_EPOCHS=4)"
cd "${VERL_DIR}"
KLSWEEP_KEEP_CLAIMS=1 SLOTS=0 SLOT_START_STAGGER_S=0 setsid nohup \
    bash "${SCRIPT_DIR}/run_constraint_only_kl_sweep_h200.sh" \
    >"${LOG_DIR}/orchestrator_slot0_17b4ep_$(date +%Y%m%d_%H%M%S).log" 2>&1 </dev/null &
sleep 180
log "claims now: $(ls "${STATE_DIR}" | tr '\n' ' ')"
