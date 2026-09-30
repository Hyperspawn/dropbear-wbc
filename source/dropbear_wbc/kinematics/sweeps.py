"""Motor sweep programs for the semantic calibration (pure numpy).

A *program* is a sequence of 22-motor position targets [rad] (motor-contract order) that one simulated
env follows stage by stage; each stage is held until the mechanism is still (see
``dropbear_wbc.isaac.quasistatic``). Only stages flagged ``record`` are measured. Programs move in
small increments (``max_step`` rad per stage) so the closed loops stay on the same assembly branch.

Two program kinds:

* ``sweep1d`` -- one motor from 0 to its lower limit (not recorded), then lower -> upper (recorded at
  every ``step``), optionally back upper -> lower (recorded, pass = 1) to expose hysteresis/jamming.
* ``grid2d`` -- one row of the two-calf-motor ankle grid: motor A to ``a`` (B held at 0), motor B to the
  first grid value, then B across the grid (recorded at grid values only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_MOTORS = 22


@dataclass
class Program:
    """Target schedule for one env."""

    name: str
    targets: list[np.ndarray] = field(default_factory=list)
    """Per-stage (22,) target vectors [rad]."""
    record: list[bool] = field(default_factory=list)
    sample: list[int] = field(default_factory=list)
    """Sample index within the program for recorded stages, -1 otherwise."""
    passno: list[int] = field(default_factory=list)
    """0 = forward pass, 1 = return pass (sweep1d only), -1 = not recorded."""

    @property
    def last(self) -> np.ndarray:
        return self.targets[-1] if self.targets else np.zeros(NUM_MOTORS)

    def __len__(self) -> int:
        return len(self.targets)

    def move_to(self, end: np.ndarray, max_step: float, record_end: bool = False, passno: int = -1) -> None:
        """Append intermediate stages from the current last target to ``end`` (inclusive)."""
        start = self.last
        delta = np.asarray(end, dtype=np.float64) - start
        n = max(1, int(np.ceil(np.max(np.abs(delta)) / max_step - 1e-9))) if np.any(delta) else 1
        for k in range(1, n + 1):
            self.targets.append(start + delta * (k / n))
            is_rec = record_end and k == n
            self.record.append(is_rec)
            self.sample.append(self._next_sample() if is_rec else -1)
            self.passno.append(passno if is_rec else -1)

    def _next_sample(self) -> int:
        return int(max([s for s in self.sample if s >= 0], default=-1) + 1)


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = max(2, int(round((hi - lo) / step)) + 1)
    return np.linspace(lo, hi, n)


def sweep1d(motor: int, lo: float, hi: float, step: float, max_step: float, base: np.ndarray | None = None,
            return_pass: bool = False, name: str | None = None) -> Program:
    """Sweep ``motor`` over [lo, hi] (rad) recording every ``step``; other motors at ``base``."""
    base = np.zeros(NUM_MOTORS) if base is None else np.asarray(base, dtype=np.float64)
    prog = Program(name=name or f"sweep1d:{motor}")
    prog.targets.append(base.copy())
    prog.record.append(False)
    prog.sample.append(-1)
    prog.passno.append(-1)
    values = _grid(lo, hi, step)
    start = base.copy()
    start[motor] = values[0]
    prog.move_to(start, max_step)
    # record the first grid point (it is the end of the unrecorded approach)
    prog.record[-1], prog.sample[-1], prog.passno[-1] = True, 0, 0
    for v in values[1:]:
        t = base.copy()
        t[motor] = v
        prog.move_to(t, max_step, record_end=True, passno=0)
    if return_pass:
        for v in values[::-1][1:]:
            t = base.copy()
            t[motor] = v
            prog.move_to(t, max_step, record_end=True, passno=1)
    return prog


def grid2d_row(motor_a: int, motor_b: int, a_value: float, b_values: np.ndarray, max_step: float,
               base: np.ndarray | None = None, name: str | None = None) -> Program:
    """One row of a 2-motor grid: A fixed at ``a_value``, B swept over ``b_values`` (recorded)."""
    base = np.zeros(NUM_MOTORS) if base is None else np.asarray(base, dtype=np.float64)
    prog = Program(name=name or f"grid2d:{motor_a}:{motor_b}:{a_value:.4f}")
    prog.targets.append(base.copy())
    prog.record.append(False)
    prog.sample.append(-1)
    prog.passno.append(-1)
    t = base.copy()
    t[motor_a] = a_value
    prog.move_to(t, max_step)
    t = t.copy()
    t[motor_b] = b_values[0]
    prog.move_to(t, max_step)
    prog.record[-1], prog.sample[-1], prog.passno[-1] = True, 0, 0
    for v in b_values[1:]:
        t = t.copy()
        t[motor_b] = v
        prog.move_to(t, max_step, record_end=True, passno=0)
    return prog


def single_pose(target: np.ndarray, max_step: float, name: str, base: np.ndarray | None = None) -> Program:
    """Move from ``base`` (default rest) to ``target`` in small steps and record the final pose."""
    base = np.zeros(NUM_MOTORS) if base is None else np.asarray(base, dtype=np.float64)
    prog = Program(name=name)
    prog.targets.append(base.copy())
    prog.record.append(False)
    prog.sample.append(-1)
    prog.passno.append(-1)
    prog.move_to(np.asarray(target, dtype=np.float64), max_step, record_end=True, passno=0)
    return prog


@dataclass
class Batch:
    """Programs packed into env slots: ``targets`` (S, N, 22), ``record`` (S, N), program id per env."""

    targets: np.ndarray
    record: np.ndarray
    sample: np.ndarray
    passno: np.ndarray
    program_ids: np.ndarray
    """(N,) index into the global program list, -1 for idle envs."""


def pack(programs: list[Program], num_envs: int) -> list[Batch]:
    """Pack programs into batches of ``num_envs`` env slots (longest first); idle slots hold rest."""
    order = sorted(range(len(programs)), key=lambda i: -len(programs[i]))
    batches: list[Batch] = []
    for start in range(0, len(order), num_envs):
        ids = order[start:start + num_envs]
        s_max = max(len(programs[i]) for i in ids)
        tg = np.zeros((s_max, num_envs, NUM_MOTORS))
        rec = np.zeros((s_max, num_envs), dtype=bool)
        smp = -np.ones((s_max, num_envs), dtype=np.int64)
        pas = -np.ones((s_max, num_envs), dtype=np.int64)
        pid = -np.ones(num_envs, dtype=np.int64)
        for slot, i in enumerate(ids):
            p = programs[i]
            n = len(p)
            tg[:n, slot] = np.stack(p.targets)
            tg[n:, slot] = p.targets[-1]
            rec[:n, slot] = p.record
            smp[:n, slot] = p.sample
            pas[:n, slot] = p.passno
            pid[slot] = i
        batches.append(Batch(tg, rec, smp, pas, pid))
    return batches


def waypoint_program(waypoints: list[np.ndarray], max_step: float, name: str, record_every: bool = False,
                     base: np.ndarray | None = None) -> Program:
    """Follow motor ``waypoints`` from ``base`` (default rest) in <= ``max_step`` increments.

    Records the final waypoint (and every waypoint if ``record_every``). Used by the verification stage to
    reach a pose along a feasible path (e.g. interpolated in semantic space) instead of a straight motor line.
    """
    base = np.zeros(NUM_MOTORS) if base is None else np.asarray(base, dtype=np.float64)
    prog = Program(name=name)
    prog.targets.append(base.copy())
    prog.record.append(False)
    prog.sample.append(-1)
    prog.passno.append(-1)
    for k, w in enumerate(waypoints):
        last = k == len(waypoints) - 1
        prog.move_to(np.asarray(w, dtype=np.float64), max_step, record_end=last or record_every, passno=0)
    return prog
