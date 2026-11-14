#!/usr/bin/env bash
# Stop run_constraint_only_kl_sweep_h200.sh, or just one of its two slots.
#
#   bash if_rlvr/exps/bidirectional/stop_constraint_only_kl_sweep_h200.sh   # orchestrator + both slots
#   SLOT=1 bash .../stop_constraint_only_kl_sweep_h200.sh                   # only slot 1's job
#   QUIET=1 SLOT=0 bash ...                                                 # between-job reap
#
# SLOT-SAFE BY CONSTRUCTION. The sweep exports KLSWEEP_SLOT=<n> into every job it launches, and
# that variable is inherited by the leaf/base launcher shells, the verl driver, the Ray head
# (gcs_server, raylet), every Ray worker, and the vLLM engine-core processes those spawn. This
# script selects processes by reading /proc/<pid>/environ for that marker (plus the slot's Ray
# temp dir /tmp/ifrlvr_r<n> in the cmdline as a backstop), so stopping slot 0 can never touch slot
# 1 -- unlike host-wide `pkill -f verl.trainer.main_ppo` / `ray::` patterns.
#
# Ordering: launcher shells first (they hold the IF_MAX_RETRIES auto-resume loop and would
# otherwise relaunch the driver), then everything else with SIGTERM, then SIGKILL. The script
# skips itself and its whole ancestor chain, which is what lets the sweep's worker call it between
# jobs without killing itself. Checkpoints are never touched; a re-run resumes from the newest.

set -euo pipefail

SLOT=${SLOT:-all}
QUIET=${QUIET:-0}

python3 - "${SLOT}" "${QUIET}" <<'PY'
import os
import signal
import sys
import time

slot, quiet = sys.argv[1], sys.argv[2] == "1"
ORCH = "run_constraint_only_kl_sweep_h200.sh"


def say(msg):
    if not quiet:
        print(f"[klsweep-stop] {msg}", flush=True)


def ancestors():
    out, pid = set(), os.getpid()
    while pid > 1:
        out.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except OSError:
            break
    return out


def read(pid, name):
    try:
        with open(f"/proc/{pid}/{name}", "rb") as f:
            return f.read()
    except OSError:
        return b""


def targets():
    skip = ancestors()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in skip:
            continue
        env = read(pid, "environ").split(b"\0")
        cmd = read(pid, "cmdline").replace(b"\0", b" ").decode(errors="replace")
        marker = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"KLSWEEP_SLOT=")), None)
        if slot == "all":
            hit = marker is not None or ORCH in cmd or "/tmp/ifrlvr_r" in cmd
        else:
            hit = marker == slot or f"/tmp/ifrlvr_r{slot}/" in cmd
        if hit:
            found.append((pid, cmd))
    return found


def kill(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


for rnd in range(4):
    procs = targets()
    if not procs:
        break
    say(f"round {rnd + 1}: {len(procs)} process(es) for slot={slot}")
    shells = [p for p, c in procs if c.startswith(("bash ", "/bin/bash ", "/usr/bin/bash ")) or ORCH in c]
    kill(shells, signal.SIGKILL)
    rest = [p for p, _ in procs if p not in shells]
    kill(rest, signal.SIGTERM)
    deadline = time.time() + 15
    while time.time() < deadline and any(os.path.exists(f"/proc/{p}") for p in rest):
        time.sleep(1)
    kill([p for p in rest if os.path.exists(f"/proc/{p}")], signal.SIGKILL)
    time.sleep(2)

left = targets()
if left:
    say(f"WARNING: {len(left)} process(es) survived: {[p for p, _ in left][:20]}")
else:
    say("clean")
PY
