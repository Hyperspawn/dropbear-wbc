"""Real-time setup and fixed-period scheduling for the wall-clock deploy loop (item 292).

A 20 ms control step that overruns its slot was invisible: ``policy_runner --clock wall`` advanced its deadline by
one period per step, so after an overrun it ran the missed steps back-to-back (a 105 ms stall gave 4 extra commands
within ~4 ms, each built from the same LowState tick) while the summary still reported ~50 Hz. :class:`WallSchedule`
drops missed slots instead of bursting and counts every late step; :func:`apply_rt` pins the control thread to a
core and asks for SCHED_FIFO so the loop is not pre-empted by ordinary load on the robot computer.
"""
from __future__ import annotations

import gc
import os
import time


def apply_rt(cpu: int | None = None, prio: int | None = None, lock_memory: bool = False,
             freeze_gc: bool = False) -> dict:
    """Best-effort RT setup for the calling process. Returns what was requested, what took effect and any errors.

    ``cpu``: pin to this core (pair with ``isolcpus=``/``nohz_full=`` on the kernel command line so nothing else is
    scheduled there). ``prio``: SCHED_FIFO priority 1..99 (needs CAP_SYS_NICE / ``ulimit -r``). ``lock_memory``:
    mlockall(MCL_CURRENT|MCL_FUTURE) so page faults cannot stall a step. ``freeze_gc``: move start-up objects out of
    the cyclic collector so later collections are short. Unsupported calls (Windows, macOS) are reported, not raised.
    """
    rep: dict = {"requested": {"cpu": cpu, "prio": prio, "lock_memory": lock_memory, "freeze_gc": freeze_gc},
                 "cpu_affinity": None, "policy": None, "prio": None, "memory_locked": False, "gc_frozen": False,
                 "errors": []}
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {int(cpu)})
        except (AttributeError, OSError, ValueError) as e:
            rep["errors"].append(f"affinity cpu {cpu}: {e!r}")
    if prio is not None:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(prio)))
        except (AttributeError, OSError, ValueError) as e:
            rep["errors"].append(f"SCHED_FIFO prio {prio}: {e!r}")
    if lock_memory:
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            if libc.mlockall(1 | 2) != 0:  # MCL_CURRENT | MCL_FUTURE
                raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
            rep["memory_locked"] = True
        except (AttributeError, OSError) as e:
            rep["errors"].append(f"mlockall: {e!r}")
    if freeze_gc and hasattr(gc, "freeze"):
        gc.collect()
        gc.freeze()
        rep["gc_frozen"] = True
    try:
        rep["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        pol = os.sched_getscheduler(0)
        rep["policy"] = {os.SCHED_OTHER: "SCHED_OTHER", os.SCHED_FIFO: "SCHED_FIFO",
                         os.SCHED_RR: "SCHED_RR"}.get(pol, str(pol))
        rep["prio"] = os.sched_getparam(0).sched_priority
    except (AttributeError, OSError):
        pass
    return rep


class WallSchedule:
    """Fixed-period deadline loop that never bursts: a step that starts more than one period late skips the missed
    slots and re-anchors on the period grid. ``late_threshold_s`` (default a quarter period) counts a step as overrun.
    """

    def __init__(self, period_s: float, t0: float | None = None, late_threshold_s: float | None = None,
                 clock=time.perf_counter, sleep=time.sleep):
        self.period = float(period_s)
        self.clock, self.sleep = clock, sleep
        self.next = clock() if t0 is None else float(t0)
        self.late_threshold = 0.25 * self.period if late_threshold_s is None else float(late_threshold_s)
        self.steps = self.overrun_steps = self.skipped_slots = 0
        self.late_s: list[float] = []

    def wait(self) -> float:
        """Sleep to the next slot; return how late (s) this step starts relative to its slot."""
        now = self.clock()
        if self.next > now:
            self.sleep(self.next - now)
            now = self.clock()
        late = max(0.0, now - self.next)  # lateness against the slot this step was due in
        if late >= self.period:
            skip = int(late // self.period)
            self.skipped_slots += skip
            self.next += skip * self.period  # re-anchor on the period grid: the next slot is in the future
        self.next += self.period
        self.steps += 1
        self.late_s.append(late)
        if late > self.late_threshold:
            self.overrun_steps += 1
        return late

    def summary(self) -> dict:
        ls = sorted(self.late_s)
        pick = (lambda q: 1e3 * ls[min(len(ls) - 1, int(q * (len(ls) - 1) + 0.5))]) if ls else (lambda q: None)
        return {"period_ms": 1e3 * self.period, "steps": self.steps, "late_threshold_ms": 1e3 * self.late_threshold,
                "overrun_steps": self.overrun_steps, "skipped_slots": self.skipped_slots,
                "late_ms": {"p50": pick(0.5), "p99": pick(0.99), "max": 1e3 * ls[-1] if ls else None}}
