#!/usr/bin/env python3
"""Watchdog for run_constraint_only_kl_sweep_h200.sh: status log + per-slot crash/hang recovery.

Runs forever next to the sweep (setsid nohup ... &). Every --poll-seconds it:

  1. writes one status line per slot to logs/kl_sweep_h200/watchdog.log
     (job, step/total, s/step over the last 5 steps, ETA, GPU memory, free disk);
  2. HANG: if a slot's job has not logged a new `step:N` line for
     max(--hang-min-minutes, --hang-step-multiple x recent s/step), it stops that slot
     (stop_constraint_only_kl_sweep_h200.sh SLOT=<s>). Its worker then marks the job failed
     and exits, which step 3 turns into a resume;
  3. DEAD SLOT: if a slot has had no process carrying KLSWEEP_SLOT=<s> for two consecutive
     polls while unfinished jobs remain, it reaps the slot, clears that slot's stale claim /
     .failed markers and starts a one-slot orchestrator
     (KLSWEEP_KEEP_CLAIMS=1 SLOTS=<s>), which re-claims the interrupted job first (the queue is
     claimed in order) and resumes it from its newest checkpoint (trainer.resume_mode=auto).

The base launcher already retries a crashed trainer up to IF_MAX_RETRIES=10 times by itself;
this covers what that loop cannot: hangs, an exhausted retry budget, a killed orchestrator.
A job is given at most --max-restarts watchdog restarts, so a deterministic failure (e.g. a
config that OOMs every time) stops instead of looping forever; the slot then stays down and
the status line says so.

Stop it with:  pkill -f kl_sweep_h200_watchdog.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

STEP_RE = re.compile(r"step:(\d+) - ")
SECONDS_RE = re.compile(r"timing_s/step:(?:np\.float64\()?([0-9.]+)")


def log(path: Path, msg: str) -> None:
    line = f"[watchdog {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with path.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def slot_pids(slot: int) -> list[int]:
    out = []
    marker = f"KLSWEEP_SLOT={slot}".encode()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as fh:
                env = fh.read().split(b"\0")
        except OSError:
            continue
        if marker in env:
            out.append(int(entry))
    return out


def claims(state_dir: Path) -> dict[str, int]:
    """job -> slot for every live claim marker."""
    result = {}
    for f in state_dir.glob("*.claimed"):
        m = re.search(r"slot=(\d+)", f.read_text(errors="replace"))
        if m:
            result[f.name[: -len(".claimed")]] = int(m.group(1))
    return result


def latest_train_log(log_dir: Path, job: str) -> Path | None:
    logs = sorted(log_dir.glob(f"{job}.train.*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def tail_steps(path: Path, max_bytes: int = 64 << 20) -> tuple[int, list[float]]:
    """(last step, recent step durations) parsed from the end of a train log."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - max_bytes))
        text = fh.read().decode(errors="replace")
    last, durations = 0, []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        last = int(m.group(1))
        d = SECONDS_RE.search(line)
        if d:
            durations.append(float(d.group(1)))
    return last, durations[-5:]


def gpu_mem(gpus: list[int]) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={','.join(map(str, gpus))}", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        return " ".join(f"{int(m) // 1024}G/{u}%" for m, u in (l.split(", ") for l in out))
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi failed: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verl-dir", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--hang-min-minutes", type=float, default=75)
    ap.add_argument("--hang-step-multiple", type=float, default=5)
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--total-steps", type=int, default=364)
    ap.add_argument("--slots", default="0 1")
    ap.add_argument("--gpus-per-slot", type=int, default=4)
    args = ap.parse_args()

    verl = Path(args.verl_dir)
    script_dir = verl / "if_rlvr/exps/bidirectional"
    log_dir = verl / "logs/kl_sweep_h200"
    state_dir = log_dir / "state"
    wlog = log_dir / "watchdog.log"
    restarts_file = state_dir / "watchdog_restarts.json"
    slots = [int(s) for s in args.slots.split()]
    jobs = os.environ.get(
        "KLSWEEP_JOBS",
        "tulu3_8b_kl0p02 tulu3_8b_kl0p05 qwen3_4b_nothink_kl0p02 qwen3_4b_nothink_kl0p05 "
        "qwen3_17b_think_kl0p02 qwen3_17b_think_kl0p05 qwen3_4b_think_kl0p02 qwen3_4b_think_kl0p05",
    ).split()

    try:
        restarts = json.loads(restarts_file.read_text())
    except (OSError, ValueError):
        restarts = {}
    absent_polls = {s: 0 for s in slots}
    progress = {}  # job -> (last_step, wallclock when it last advanced)
    given_up = set()

    log(wlog, f"started; poll={args.poll_seconds}s hang>=max({args.hang_min_minutes}min, "
              f"{args.hang_step_multiple}x s/step) max_restarts={args.max_restarts}")

    while True:
        try:
            now = time.time()
            live = claims(state_dir)
            done = {j for j in jobs if (state_dir / f"{j}.done").exists()}
            failed = {j for j in jobs if (state_dir / f"{j}.failed").exists()}
            remaining = [j for j in jobs if j not in done]
            disk = shutil.disk_usage(verl).free // 2**30
            if not remaining:
                log(wlog, f"all {len(jobs)} jobs done; exiting")
                return 0

            for slot in slots:
                gpus = list(range(slot * args.gpus_per_slot, (slot + 1) * args.gpus_per_slot))
                job = next((j for j, s in live.items() if s == slot), None)
                pids = slot_pids(slot)
                status = f"slot{slot} gpus[{gpu_mem(gpus)}] disk={disk}G "

                if job:
                    path = latest_train_log(log_dir, job)
                    step, durs = tail_steps(path) if path else (0, [])
                    prev = progress.get(job)
                    if prev is None or step != prev[0]:
                        progress[job] = (step, now)
                    stalled_min = (now - progress[job][1]) / 60
                    sps = sum(durs) / len(durs) if durs else 0.0
                    eta_h = (args.total_steps - step) * sps / 3600 if sps else float("nan")
                    status += (f"{job} step {step}/{args.total_steps} {sps:.0f}s/step ETA {eta_h:.1f}h "
                               f"(no new step for {stalled_min:.0f}min)")
                    hang_after = max(args.hang_min_minutes, args.hang_step_multiple * sps / 60)
                    if pids and stalled_min > hang_after:
                        log(wlog, f"HANG slot{slot} {job}: no new step for {stalled_min:.0f}min "
                                  f"(> {hang_after:.0f}); stopping the slot so it is resumed")
                        subprocess.run(["bash", str(script_dir / "stop_constraint_only_kl_sweep_h200.sh")],
                                       env={**os.environ, "SLOT": str(slot), "QUIET": "1"}, timeout=600)
                        progress.pop(job, None)
                else:
                    status += "idle"
                log(wlog, status + ("" if pids else " [no slot processes]"))

                # ---- dead slot -> resume it ----
                absent_polls[slot] = 0 if pids else absent_polls[slot] + 1
                # A live worker always holds a claim, and between two jobs it can briefly have no
                # marked process (reap + wait for GPUs, < ~6 min). With a claim present, require
                # 3 empty polls (15 min) so that gap is never mistaken for a dead slot; without a
                # claim the worker has exited (it removes its claim when it marks a job failed).
                if absent_polls[slot] < (3 if job else 2):
                    continue
                busy_elsewhere = {j for j, s in live.items() if s != slot}
                todo = [j for j in remaining if j not in busy_elsewhere]
                if not todo:
                    continue
                nxt = todo[0]
                if nxt in given_up:
                    continue
                if restarts.get(nxt, 0) >= args.max_restarts:
                    given_up.add(nxt)
                    log(wlog, f"GIVE UP slot{slot}: {nxt} already restarted {restarts[nxt]}x; "
                              f"leaving the slot down for manual inspection")
                    continue
                restarts[nxt] = restarts.get(nxt, 0) + 1
                restarts_file.write_text(json.dumps(restarts, indent=2))
                log(wlog, f"RESTART slot{slot}: resuming {nxt} (watchdog restart "
                          f"{restarts[nxt]}/{args.max_restarts}; was failed={nxt in failed})")
                subprocess.run(["bash", str(script_dir / "stop_constraint_only_kl_sweep_h200.sh")],
                               env={**os.environ, "SLOT": str(slot), "QUIET": "1"}, timeout=600)
                for j, s in live.items():
                    if s == slot:
                        (state_dir / f"{j}.claimed").unlink(missing_ok=True)
                (state_dir / f"{nxt}.failed").unlink(missing_ok=True)
                out = open(log_dir / f"orchestrator_slot{slot}_restart{int(now)}.log", "w")
                subprocess.Popen(
                    ["bash", str(script_dir / "run_constraint_only_kl_sweep_h200.sh")],
                    cwd=verl, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env={**os.environ, "KLSWEEP_KEEP_CLAIMS": "1", "SLOTS": str(slot),
                         "SLOT_START_STAGGER_S": "0"},
                    start_new_session=True,
                )
                absent_polls[slot] = 0
        except Exception as exc:  # noqa: BLE001 - the watchdog must outlive any single failure
            log(wlog, f"poll error: {exc!r}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
