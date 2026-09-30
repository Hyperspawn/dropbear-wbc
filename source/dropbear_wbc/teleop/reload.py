"""Non-blocking calibration hot reload for the teleop control loop (review fix 2026-09-24).

``DropbearArmIK._load`` rebuilds the measured elbow tables through ``CalibrationFK`` (about 0.5 s on this machine).
Calling ``reload_if_changed()`` inside the 50 Hz loop therefore stalled the LowCmd stream for longer than the
bridge's 100 ms watchdog (CONTRACTS 6.1), which dropped every motor into damping. :class:`BackgroundReloader` watches
the calibration file on a daemon thread, builds the replacement objects there and hands them to the loop, which
swaps them in atomically between two control steps (:meth:`BackgroundReloader.take` never blocks).
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class BackgroundReloader:
    """Watch ``path``; when its SHA-256 changes, call ``factory()`` on a background thread.

    Args:
        path: file to watch (the calibration JSON).
        factory: builds the replacement (e.g. ``(DropbearArmIK(...), ArmGravity(...))``); runs on the watcher thread.
        current_sha256: SHA-256 of the file the current objects were built from (no reload for it).
        poll_s: polling period [s].
    """

    def __init__(self, path: str | Path, factory: Callable[[], Any], current_sha256: str | None, poll_s: float = 0.5):
        self.path = Path(path)
        self.factory = factory
        self.sha256 = current_sha256
        self.poll_s = float(poll_s)
        self._pending: tuple[str, Any, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.errors: list[str] = []
        self.builds: list[dict] = []
        self._thread = threading.Thread(target=self._run, name="calibration-reloader", daemon=True)

    def start(self) -> "BackgroundReloader":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        last_mtime = None
        while not self._stop.wait(self.poll_s):
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                continue
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            sha = _sha256(self.path)
            if sha is None or sha == self.sha256:
                continue
            t0 = time.perf_counter()
            try:
                obj = self.factory()
            except Exception as exc:  # noqa: BLE001 - keep the old objects; report
                self.errors.append(f"{type(exc).__name__}: {exc}")
                continue
            build_s = time.perf_counter() - t0
            with self._lock:
                self._pending = (sha, obj, build_s)
            self.sha256 = sha
            self.builds.append({"sha256": sha, "build_s": build_s})

    def take(self) -> tuple[str, Any, float] | None:
        """``(sha256, built objects, build seconds)`` if a new build is ready, else ``None``. Never blocks."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            out, self._pending = self._pending, None
            return out
        finally:
            self._lock.release()
