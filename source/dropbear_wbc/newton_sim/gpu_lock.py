"""GPU lock protocol from CONTRACTS section 0 (one Isaac Sim / Newton GPU process at a time).

The lock is ``<repo>/.locks/gpu.lock`` created exclusively; it holds the owner
name, PID and start time as JSON. Waiters poll every 30 s for up to 60 min.
Always release in a ``finally`` block (the context manager does this).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / ".locks" / "gpu.lock"


class GpuLockTimeout(TimeoutError):
    """Raised when the lock could not be acquired within the timeout."""


class GpuLock:
    """Exclusive-create file lock.

    Args:
        owner: human-readable owner name written into the lock file.
        path: lock file path.
        poll_s: seconds between acquisition attempts.
        timeout_s: give up after this many seconds.
    """

    def __init__(self, owner: str, path: Path = DEFAULT_LOCK, poll_s: float = 30.0, timeout_s: float = 3600.0):
        self.owner, self.path, self.poll_s, self.timeout_s = owner, Path(path), poll_s, timeout_s
        self.held = False
        self.waited_s = 0.0

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                waited = time.monotonic() - start
                if waited > self.timeout_s:
                    raise GpuLockTimeout(f"GPU lock {self.path} still held after {waited:.0f} s: {self._holder()}")
                print(f"[gpu_lock] held by {self._holder()}; waiting {self.poll_s:.0f} s "
                      f"(waited {waited:.0f} s)", file=sys.stderr, flush=True)
                time.sleep(self.poll_s)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"owner": self.owner, "pid": os.getpid(), "host": socket.gethostname(),
                           "start_time": datetime.now(timezone.utc).isoformat()}, f)
            self.held = True
            self.waited_s = time.monotonic() - start
            return

    def _holder(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return "<unreadable>"

    def release(self) -> None:
        """Delete the lock if this process holds it (never deletes someone else's lock)."""
        if not self.held:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid():
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.held = False

    def __enter__(self) -> "GpuLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
