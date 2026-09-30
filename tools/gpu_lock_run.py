"""Run one command under the team GPU lock (``.locks/gpu.lock``) and tee its output to a log file.

Contract section 0 (GPU lock): the lock file is created exclusively and holds owner, PID and start time;
if it exists we poll every 30 s for up to 60 min; it is always deleted in ``finally`` (only if it is
still ours). Stdlib only, runs with any Python 3.10+.

Usage (from ``<repo>``)::

    python tools/gpu_lock_run.py --owner robot_task --log logs/robot_task/x.log --timeout 1800 -- \
        C:/isaac-sim/python.bat -u scripts/train.py --headless ...

Exit code: the child's exit code; 124 on timeout; 125 if the lock could not be acquired.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / ".locks" / "gpu.lock"


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def acquire(owner: str, command: list[str], wait_minutes: float, poll_s: float = 30.0) -> str:
    """Create the lock exclusively; return our token. Raises TimeoutError after ``wait_minutes``."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    payload = json.dumps(
        {"owner": owner, "pid": os.getpid(), "start_time": _now(), "token": token, "command": command}, indent=1
    )
    deadline = time.monotonic() + wait_minutes * 60.0
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = LOCK_PATH.read_text(encoding="utf-8")
            except OSError:
                holder = "<unreadable>"
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GPU lock still held after {wait_minutes} min: {holder}")
            print(f"[gpu_lock_run] {_now()} lock busy, holder={holder.strip()!r}; retry in {poll_s:.0f}s", flush=True)
            time.sleep(poll_s)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        return token


def release(token: str) -> None:
    """Delete the lock if it still carries our token."""
    try:
        content = LOCK_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    if token in content:
        LOCK_PATH.unlink(missing_ok=True)
    else:
        print(f"[gpu_lock_run] lock no longer ours, leaving it: {content!r}", flush=True)


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", default="robot_task")
    parser.add_argument("--log", type=Path, required=True, help="log file (relative to repo root or absolute)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="child wall-clock timeout [s]")
    parser.add_argument("--wait-minutes", type=float, default=60.0, help="max wait for the lock [min]")
    parser.add_argument("--drain-s", type=float, default=10.0,
                        help="after the child exits, wait at most this long for its output pipe to close [s]")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given (put it after --)")
    log_path = args.log if args.log.is_absolute() else REPO_ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        token = acquire(args.owner, command, args.wait_minutes)
    except TimeoutError as exc:
        print(f"[gpu_lock_run] {exc}", flush=True)
        return 125
    rc = 1
    start = time.monotonic()
    proc: subprocess.Popen | None = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            header = f"[gpu_lock_run] start={_now()} owner={args.owner} cwd={REPO_ROOT}\n[gpu_lock_run] cmd={command}\n"
            log.write(header)
            log.flush()
            sys.stdout.write(header)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            timed_out = threading.Event()

            def _watchdog() -> None:
                if proc is not None and proc.poll() is None:
                    timed_out.set()
                    _kill_tree(proc)

            timer = threading.Timer(args.timeout, _watchdog)
            timer.daemon = True
            timer.start()
            assert proc.stdout is not None
            write_lock = threading.Lock()

            def _pump() -> None:
                # Reads until EOF. EOF may never come: Kit starts the Omniverse hub (hub.exe --mode=shared), which
                # inherits this pipe and outlives the child (2026-09-24, logs/gpu_pipeline/PROGRESS.md). So the pump
                # is a daemon thread and the lock lifetime follows the child PROCESS, not the pipe.
                for line in proc.stdout:
                    with write_lock:
                        log.write(line)
                        log.flush()
                        sys.stdout.write(line)

            pump = threading.Thread(target=_pump, daemon=True)
            pump.start()
            proc.wait()
            timer.cancel()
            pump.join(timeout=args.drain_s)
            drained = not pump.is_alive()
            rc = 124 if timed_out.is_set() else int(proc.returncode)
            footer = (
                f"[gpu_lock_run] end={_now()} rc={rc} elapsed_s={time.monotonic() - start:.1f}"
                f"{' (TIMEOUT)' if timed_out.is_set() else ''}"
                f"{'' if drained else ' (stdout pipe still held by a detached descendant, e.g. hub.exe; not waiting)'}\n"
            )
            with write_lock:
                log.write(footer)
                log.flush()
                sys.stdout.write(footer)
    finally:
        if proc is not None:
            _kill_tree(proc)
        release(token)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
