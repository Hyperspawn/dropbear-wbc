"""Local overnight watchdog for the Brev training box (Windows host, runs detached).

* Keeps the laptop from idle-sleeping while it runs (SetThreadExecutionState; a closed lid still sleeps).
* Every PULL_EVERY_S: rsync logs + checkpoints from the instance to logs/brev/remote/ (via WSL ssh + brev ssh config).
* At the budget DEADLINE (or when all four GPU queues report QUEUE_DONE, or on logs/brev/STOP_AND_DELETE):
  two final pulls, then `brev delete <instance>` (retried until `brev ls` no longer lists it). This is what stops
  the billing: the user approved ~$80 (2026-09-25 night), ~$90 total, then added $50 (16:30 IST) -> cap $130.
* Overrides without restarting: logs/brev/deadline.txt (ISO time) replaces the deadline; logs/brev/budget.txt (USD)
  replaces the hard budget cap.
* Every action is appended to logs/brev/watchdog.log (JSON lines).
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BREV_DIR = REPO / "logs" / "brev"
LOG = BREV_DIR / "watchdog.log"
INSTANCE = "dropbear-train"
PRICE_PER_H = 4.66
# session 2 (2026-09-26): new instance after the user added $50; the real balance is uncertain (the powered-off
# session-1 VM may have billed until it disappeared), so the cap is $35 (about 7.5 h)
CREATED = dt.datetime(2026, 9, 26, 10, 43, 40, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
DEFAULT_DEADLINE = dt.datetime(2026, 9, 26, 18, 10, tzinfo=CREATED.tzinfo)
HARD_BUDGET_USD = 35.0                                                      # delete regardless beyond this
PULL_EVERY_S = 30 * 60
WSL_HOME = "$HOME"  # expanded by the WSL login shell (commands run via bash -lc)
SSH = f"ssh -F {WSL_HOME}/.brev/ssh_config -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=30"
BREV = f"{WSL_HOME}/.local/bin/brev"
# this repository as seen from WSL (H:\\x -> /mnt/h/x)
DEST = f"/mnt/{REPO.drive[0].lower()}{REPO.as_posix()[2:]}/logs/brev/remote" if REPO.drive else str(REPO / "logs/brev/remote")

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def now() -> dt.datetime:
    return dt.datetime.now(CREATED.tzinfo)


def log(**kw) -> None:
    kw = {"time": now().isoformat(timespec="seconds"), **kw}
    BREV_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")


def wsl(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(["wsl", "-d", os.environ.get("WSL_DISTRO", "Ubuntu"), "--", "bash", "-lc", cmd], capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def deadline() -> dt.datetime:
    f = BREV_DIR / "deadline.txt"
    if f.exists():
        try:
            d = dt.datetime.fromisoformat(f.read_text().strip())
            return d if d.tzinfo else d.replace(tzinfo=CREATED.tzinfo)
        except ValueError:
            pass
    return DEFAULT_DEADLINE


def hard_budget() -> float:
    f = BREV_DIR / "budget.txt"
    if f.exists():
        try:
            return float(f.read_text().strip())
        except ValueError:
            pass
    return HARD_BUDGET_USD


def pull() -> bool:
    inc = ("--include='*/' --include='logs/brev/**' --include='logs/rsl_rl/**' --include='*.log' "
           "--exclude='*'")
    ok = True
    for g in range(4):
        rc, out = wsl(f"mkdir -p {DEST}/dbw{g} && rsync -az --partial --prune-empty-dirs {inc} -e \"{SSH}\" "
                      f"{INSTANCE}:~/dbw{g}/ {DEST}/dbw{g}/")
        ok &= rc == 0
        if rc != 0:
            log(event="pull_fail", gpu=g, rc=rc, out=out[-600:])
    rc, out = wsl(f"mkdir -p {DEST}/runs && rsync -az -e \"{SSH}\" {INSTANCE}:~/runs/ {DEST}/runs/ ; "
                  f"rsync -az -e \"{SSH}\" {INSTANCE}:~/setup_instance.log {DEST}/ ; "
                  f"{SSH} {INSTANCE} 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader' "
                  f"> {DEST}/nvidia_smi_last.txt 2>&1")
    log(event="pull", ok=ok)
    return ok


def queues_done() -> bool:
    runs = REPO / "logs" / "brev" / "remote" / "runs"
    try:
        return all("QUEUE_DONE" in (runs / f"gpu{g}.log").read_text() for g in range(4))
    except OSError:
        return False


def delete_instance(reason: str) -> None:
    log(event="final_pull_start", reason=reason)
    pull()
    pull()
    for attempt in range(60):
        rc, out = wsl(f"{BREV} delete {INSTANCE} --no-check-latest < /dev/null", timeout=300)
        log(event="delete_attempt", attempt=attempt, rc=rc, out=out[-400:])
        if rc != 0 and attempt == 3:
            # 2026-09-26 lesson: an expired brev login makes every delete fail; SSH (keys) still works, so stop the
            # jobs and power the VM off while retrying (billing of a powered-off VM depends on the provider)
            rc3, out3 = wsl(f"{SSH} {INSTANCE} 'for g in 0 1 2 3; do tmux kill-session -t gpu$g; done; "
                            f"pkill -f run_chunked; sudo -n shutdown -h +0' < /dev/null", timeout=120)
            log(event="delete_failing_ssh_poweroff", rc=rc3, out=out3[-300:])
        time.sleep(60)
        rc2, ls = wsl(f"{BREV} ls --no-check-latest < /dev/null", timeout=120)
        if rc2 == 0 and INSTANCE not in ls:
            log(event="deleted", spent_est_usd=round((now() - CREATED).total_seconds() / 3600 * PRICE_PER_H, 2))
            return
    log(event="DELETE_FAILED_GIVE_UP")


def main() -> int:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log(event="start", deadline=deadline().isoformat(), pid=__import__("os").getpid())
    last_pull = 0.0
    while True:
        spent = (now() - CREATED).total_seconds() / 3600 * PRICE_PER_H
        reason = None
        if (BREV_DIR / "STOP_AND_DELETE").exists():
            reason = "stop_file"
        elif now() >= deadline():
            reason = "deadline"
        elif spent >= hard_budget():
            reason = f"budget {spent:.2f}"
        elif queues_done():
            reason = "all_queues_done"
        if reason:
            delete_instance(reason)
            return 0
        if time.time() - last_pull >= PULL_EVERY_S:
            pull()
            last_pull = time.time()
            log(event="status", spent_est_usd=round(spent, 2), deadline=deadline().isoformat())
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
