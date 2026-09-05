#!/usr/bin/env bash
# Stop the Qwen3-1.7B KL-coefficient ablation cleanly (and reap anything it left behind).
#
# WHY THIS EXISTS. Killing the trainer process is NOT enough and actively makes things worse:
# qwen3_4b_01_00_const1_ref_anchor_reasoning.sh -- the launcher every arm ends up in -- wraps
# `python3 -m verl.trainer.main_ppo` in an auto-resume loop (IF_MAX_RETRIES, default 10). Kill only
# the driver and that loop sees a non-zero rc and relaunches it. If a second arm has meanwhile
# started, two trainers share the same 8 GPUs, Ray temp dir, namespace and worker port range, which
# surfaces as a confusing Gloo error that names none of the real cause:
#
#     RuntimeError: [.../gloo/transport/tcp/pair.cc:544] Connection closed by peer
#     ... in _build_model_optimizer -> torch.distributed.barrier()
#
# So: wrappers first (they hold the retry loop), then the driver, then vLLM, then Ray.
#
# Also note `pkill -f <pattern>` matches pkill's OWN command line when the pattern appears in it,
# so a naive one-liner can kill the shell running it and leave the teardown half-done. This script
# matches on /proc/<pid>/cmdline and skips itself AND its whole ancestor chain -- which is what
# lets run_qwen3_17b_kl_ablation.sh call it between arms without killing the sweep.
#
#   bash if_rlvr/exps/bidirectional/stop_qwen3_17b_kl_ablation.sh        # stop everything
#   CLEAR_RAY_TMPDIR=1 bash .../stop_qwen3_17b_kl_ablation.sh            # also drop /tmp/ifrlvr_r*
#   QUIET=1 bash .../stop_qwen3_17b_kl_ablation.sh                       # between-arm reap
#
# Checkpoints are left untouched; trainer.resume_mode=auto picks up from the newest one.
#
# NOTE: the `verl.trainer.main_ppo` and `ray::` patterns are host-wide. If you ever run an
# unrelated verl job on this box at the same time, this will take it down too.

set -euo pipefail

RUN_SLOT=${RUN_SLOT:-0}
CLEAR_RAY_TMPDIR=${CLEAR_RAY_TMPDIR:-0}
QUIET=${QUIET:-0}

python3 - "${QUIET}" <<'PY'
import os
import re
import signal
import sys
import time

quiet = sys.argv[1] == "1"


def say(msg):
    if not quiet:
        print(msg, flush=True)


def ancestors():
    """Every pid from this process up to init. Excluding the whole chain (not just getppid())
    is what makes this safe to call from inside the sweep runner, which sits several bash
    frames above us and matches the wrapper tier's own pattern."""
    chain, pid = set(), os.getpid()
    while pid and pid not in chain:
        chain.add(pid)
        try:
            with open(f"/proc/{pid}/stat", "rb") as handle:
                # comm can contain spaces and parens; PPID is the field after the final ')'.
                stat = handle.read().decode(errors="replace")
            pid = int(stat[stat.rindex(")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return chain


PROTECTED = ancestors()

# Ordered: anything that can respawn a trainer dies before the trainer does.
TIERS = [
    (
        "wrapper/retry-loop",
        [
            r"run_qwen3_17b_kl_ablation\.sh",
            r"qwen3_4b_01_00_const1_ref_anchor_reasoning\.sh",
        ],
    ),
    ("tee", [r"tee .*qwen3_17b_kl_ablation"]),
    ("trainer driver", [r"verl\.trainer\.main_ppo"]),
    ("vllm engine", [r"vLLMHttpServer", r"EngineCore"]),
    (
        "ray",
        [
            r"ray::",
            r"/raylet ",
            r"gcs_server",
            r"plasma_store",
            r"ray/dashboard/agent\.py",
            r"runtime_env/agent/main\.py",
            r"default_worker\.py",
        ],
    ),
]
ALL_PATTERNS = [p for _, pats in TIERS for p in pats]


def scan(patterns):
    """Fresh /proc scan. Re-reading matters: a pre-kill snapshot still lists pids whose /proc
    entry lingers for a moment after SIGKILL, which would report phantom survivors."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in PROTECTED:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                cmd = handle.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(re.search(p, cmd) for p in patterns):
            out.append((pid, cmd))
    return out


for label, patterns in TIERS:
    hits = scan(patterns)
    for pid, _ in hits:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    say(f"[stop] {label:20s} killed {len(hits):4d} pid(s)")

# Ray can respawn a worker or two while its raylet is dying; sweep until the box is quiet.
survivors = []
for attempt in range(4):
    time.sleep(1.0)
    survivors = scan(ALL_PATTERNS)
    if not survivors:
        break
    for pid, _ in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    say(f"[stop] sweep {attempt + 1}: re-killed {len(survivors)} lingering pid(s)")

if survivors:
    print(f"[stop] WARNING: {len(survivors)} process(es) still present after 4 sweeps:", flush=True)
    for pid, cmd in survivors[:10]:
        print(f"[stop]   {pid} {cmd[:100]}", flush=True)
else:
    say("[stop] no run/ray processes remain")
PY

if [[ "${CLEAR_RAY_TMPDIR}" == "1" ]]; then
    rm -rf "/tmp/ifrlvr_r${RUN_SLOT}"
    [[ "${QUIET}" == "1" ]] || echo "[stop] cleared /tmp/ifrlvr_r${RUN_SLOT}"
fi

if [[ "${QUIET}" != "1" ]]; then
    echo "[stop] GPU state:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/[stop]   /'
fi
