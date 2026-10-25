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


class StopRequest:
    """Records why a LowCmd writer stopped, so a signal stop is not reported as a clean, completed run.

    ``handler`` is the callable for :func:`install_stop_handler`. ``request(cause)`` marks an internal stop
    (``duration``, ``device_stop``). The first cause wins; later signals only bump ``count``.
    """

    def __init__(self) -> None:
        self.flag = False
        self.count = 0
        self.cause: str | None = None
        self.signum: int | None = None
        self.signals: list[str] = []

    def handler(self, signum, _frame=None) -> None:
        name = signal.Signals(signum).name
        self.signals.append(name)
        self.count += 1
        if self.cause is None:
            self.cause, self.signum = name, int(signum)
        self.flag = True

    def request(self, cause: str) -> None:
        if self.cause is None:
            self.cause = cause
        self.flag = True

    @property
    def by_signal(self) -> bool:
        return self.signum is not None

    def exit_code(self) -> int:
        """Shell convention for a process stopped by a signal: 128 + signum (SIGINT 130, SIGTERM 143, SIGHUP 129)."""
        return 128 + self.signum if self.signum is not None else 0

    def to_dict(self) -> dict:
        return {"cause": self.cause, "by_signal": self.by_signal, "signals": list(self.signals), "count": self.count}
