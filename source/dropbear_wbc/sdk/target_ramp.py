"""Motor-side position-target ramp: the robot-side twin of ``DatasheetMotor.target_interp_steps``.

Policies trained with an ``hw_*i`` profile (e.g. ``hw_v1i``) saw every new 50 Hz position target ramped linearly
over N physics steps (5 ms each) instead of a step (docs/ISSUES.md #11: a third fewer 200 Hz torque jumps). The
deploy FSM runs at 50 Hz and cannot do it; the loop that talks to the motors must: ``tools/newton_bridge.py
--target-ramp-ms`` in sim2sim, the ESP32 / motor firmware on the robot. Same law as the sim:

    on a new target: from = the target currently applied, to = new, k = 0
    every control tick: k = min(k + 1, n); applied = from + (k / n) * (to - from)

``snap`` (first command, leaving damping) jumps straight to the new target, like a reset env in the sim.
"""
from __future__ import annotations

import numpy as np


class TargetRamp:
    def __init__(self, n_ticks: int):
        self.n = max(1, int(n_ticks))
        self.k = self.n
        self.src: np.ndarray | None = None
        self.dst: np.ndarray | None = None
        self.applied: np.ndarray | None = None

    @classmethod
    def from_ms(cls, ramp_ms: float, tick_dt_s: float) -> "TargetRamp":
        """A ramp of ``ramp_ms`` at a control tick of ``tick_dt_s`` (0 ms = zero-order hold)."""
        return cls(int(round(ramp_ms * 1e-3 / tick_dt_s)) if ramp_ms > 0 else 1)

    @property
    def active(self) -> bool:
        return self.k < self.n

    def on_command(self, q: np.ndarray, snap: bool = False) -> np.ndarray:
        """Register a new target; returns the target to apply NOW (the ramp start, or ``q`` on a snap / no ramp)."""
        q = np.asarray(q, dtype=float).copy()
        if snap or self.applied is None or self.n <= 1:
            self.src, self.dst, self.k, self.applied = q.copy(), q.copy(), self.n, q.copy()
            return self.applied
        if self.dst is not None and np.array_equal(q, self.dst):
            return self.applied  # a repeated command does not restart the ramp (the sim ramps only NEW targets)
        self.src, self.dst, self.k = self.applied.copy(), q, 0
        return self.applied

    def tick(self) -> np.ndarray | None:
        """Advance one control tick; returns the new target to apply, or ``None`` when nothing changes."""
        if not self.active or self.src is None or self.dst is None:
            return None
        self.k += 1
        self.applied = self.src + (self.k / self.n) * (self.dst - self.src)
        return self.applied

    def cancel(self) -> None:
        """Stop ramping (e.g. the watchdog dropped to damping); the next command snaps."""
        self.k, self.applied = self.n, None
