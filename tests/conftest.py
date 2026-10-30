"""Host-wide lock around the fixed-port loopback tests.

The loopback tests bind fixed 127.0.0.1 ports (see test_loopback_port_plan.py), so two pytest runs on
one host (for example two venvs, or a local run next to an editor's test runner) collide with
EADDRINUSE. Every test in a module that hands fixed ports to Endpoints holds an exclusive file lock in
the system temp dir for its duration; a concurrent run waits instead of failing. Tests that use
OS-assigned ports do not take the lock.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pytest

LOCK_PATH = Path(tempfile.gettempdir()) / "dropbear_wbc_loopback_ports.lock"
LOCK_TIMEOUT_S = float(os.environ.get("DROPBEAR_LOOPBACK_LOCK_TIMEOUT_S", "600"))
_FIXED_PORT_RE = re.compile(r"\b(?:cmd_port|state_port)\s*=\s*\d{4,5}|\"SIG[A-Z]+\",\s*\d{4,5}\)")
_uses_fixed_ports: dict[str, bool] = {}


def module_uses_fixed_ports(path: str | os.PathLike) -> bool:
    key = str(path)
    if key not in _uses_fixed_ports:
        text = Path(path).read_text(encoding="utf-8")
        _uses_fixed_ports[key] = "Endpoints(" in text and bool(_FIXED_PORT_RE.search(text))
    return _uses_fixed_ports[key]


class HostLock:
    """Exclusive inter-process lock on a file (fcntl on POSIX, msvcrt on Windows)."""

    def __init__(self, path: Path, timeout_s: float):
        self.path, self.timeout_s, self._fh = path, timeout_s, None

    def acquire(self) -> float:
        t0 = time.monotonic()
        fh = open(self.path, "a+b")
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fh
                return time.monotonic() - t0
            except OSError:
                if time.monotonic() - t0 > self.timeout_s:
                    fh.close()
                    raise TimeoutError(f"loopback port lock {self.path} held for > {self.timeout_s:.0f} s")
                time.sleep(0.05)

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


@pytest.fixture(autouse=True)
def _loopback_port_lock(request):
    path = getattr(request.module, "__file__", None)
    if not path or not module_uses_fixed_ports(path):
        yield
        return
    lock = HostLock(LOCK_PATH, LOCK_TIMEOUT_S)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
