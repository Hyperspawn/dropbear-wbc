"""Stop signals for LowCmd writers.

Every process that publishes ``rt/lowcmd`` must leave the robot in damping (kp 0) when it is asked to stop. Python
only turns SIGINT into a catchable event by default; SIGTERM (``kill``, systemd/docker stop, ``Popen.terminate``) and
SIGHUP (SSH session dropped) kill the interpreter without running ``finally`` blocks, so the last LowCmd (hold or
policy gains) stays on the robot. ``install_stop_handler`` routes all of them to the same graceful-stop handler.
"""
from __future__ import annotations

import signal
from typing import Callable

STOP_SIGNALS: tuple[signal.Signals, ...] = tuple(
    getattr(signal, n) for n in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, n)
)


def install_stop_handler(handler: Callable = signal.default_int_handler) -> tuple[signal.Signals, ...]:
    """Register ``handler`` for SIGINT, SIGTERM and (POSIX) SIGHUP. The default handler raises KeyboardInterrupt,
    so ``except KeyboardInterrupt`` / ``finally`` damping paths also run on SIGTERM and SIGHUP. Main thread only."""
    for s in STOP_SIGNALS:
        signal.signal(s, handler)
    return STOP_SIGNALS
