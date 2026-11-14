#!/usr/bin/env bash
# One-shot helper: once <job> has saved AND pushed global_step_<STEP>, stop its slot and resume the
# job from that checkpoint with the current orchestrator code (used to switch the running Tulu
# jobs onto VERL_SKIP_OLD_LOGPROB_ON_POLICY=1 at step 182 without losing any progress).
#
#   setsid nohup bash kl_sweep_restart_at_step.sh <slot> <job> <step> >>log 2>&1 &
set -euo pipefail
SLOT=$1 JOB=$2 STEP=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LOG_DIR=${VERL_DIR}/logs/kl_sweep_h200
STATE_DIR=${LOG_DIR}/state
PUSH_LOG=${LOG_DIR}/${JOB}.push.log
log() { echo "[restart-at-step $(date '+%F %T')] slot${SLOT} ${JOB}: $*"; }

log "waiting for global_step_${STEP} to be uploaded (${PUSH_LOG})"
until grep -q "uploaded global_step_${STEP} " "${PUSH_LOG}" 2>/dev/null; do
    if [[ -e "${STATE_DIR}/${JOB}.done" || -e "${STATE_DIR}/${JOB}.failed" ]]; then
        log "job is already done/failed; nothing to do"; exit 0
    fi
    sleep 30
done
log "global_step_${STEP} is on the Hub; pausing the watchdog and stopping slot ${SLOT}"
# Freeze the watchdog so it does not race this restart with its own dead-slot recovery.
pkill -STOP -f kl_sweep_h200_watchdog.py || true
trap 'pkill -CONT -f kl_sweep_h200_watchdog.py || true' EXIT
SLOT=${SLOT} QUIET=1 bash "${SCRIPT_DIR}/stop_constraint_only_kl_sweep_h200.sh" || true

# The slot's worker survives the stop: it pushes once more, verifies, reaps the GPUs, then marks the
# job .failed, drops its claim and exits. Wait for that before relaunching.
for _ in $(seq 1 120); do
    [[ -e "${STATE_DIR}/${JOB}.failed" && ! -e "${STATE_DIR}/${JOB}.claimed" ]] && break
    sleep 10
done
if [[ ! -e "${STATE_DIR}/${JOB}.failed" ]]; then
    log "worker did not mark the job failed within 20 min; clearing its claim anyway"
fi
# kill any leftover orchestrator bash for this slot (its environ has no KLSWEEP_SLOT marker)
for pid in $(pgrep -f run_constraint_only_kl_sweep_h200.sh || true); do
    if tr '\0' '\n' </proc/${pid}/environ 2>/dev/null | grep -qx "SLOTS=${SLOT}"; then
        kill "${pid}" 2>/dev/null || true
    fi
done
rm -f "${STATE_DIR}/${JOB}.claimed" "${STATE_DIR}/${JOB}.failed"

log "relaunching slot ${SLOT} (resumes ${JOB} from global_step_${STEP})"
cd "${VERL_DIR}"
KLSWEEP_KEEP_CLAIMS=1 SLOTS=${SLOT} SLOT_START_STAGGER_S=0 setsid nohup \
    bash "${SCRIPT_DIR}/run_constraint_only_kl_sweep_h200.sh" \
    >"${LOG_DIR}/orchestrator_slot${SLOT}_skipoldlp_$(date +%Y%m%d_%H%M%S).log" 2>&1 </dev/null &
sleep 180
log "claims now: $(ls "${STATE_DIR}" | tr '\n' ' ')"
