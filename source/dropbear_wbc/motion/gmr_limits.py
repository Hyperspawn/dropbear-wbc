"""A mink limit that keeps GMR's IK inside Dropbear's MOTOR limits while it solves on the serial model.

The DERIVED serial model's joint ranges are the semantic valid ranges, which for the ``serial3`` hips/shoulders
and the ``lut2d`` ankles are bounding boxes: many (pitch, roll, yaw) / (pitch, roll) combinations inside the box
need motor angles outside the motor limits, and ``SemanticMap.semantic_to_motor`` then clips them afterwards
(moving the feet). :class:`DropbearMotorLimit` adds, at every IK iteration, the linearised motor-space bounds

    m_lo <= m(s) + J(s) ds <= m_hi,        J = dm/ds (finite differences of the calibration's own inverse maps)

for the chain motors of the four ``serial3`` groups and the two calf-motor pairs, so the IK redistributes the
error instead of the post-hoc clip. ``m(s)`` uses UNCLIPPED inverses (serial3 Newton; the ankle's bilinear table
extrapolates linearly outside the swept grid), so a violated bound pushes back. Standard mink ``Limit`` interface
(``G dq <= h``, dq = tangent-space displacement), usable with GMR by appending it to ``GeneralMotionRetargeting.ik_limits``.
"""
from __future__ import annotations

import numpy as np

from dropbear_wbc.kinematics.semantic import SEMANTIC_NAMES, SemanticMap, _bilinear

try:  # mink is only present in the GMR venv
    from mink.limits.limit import Constraint, Limit
except Exception:  # pragma: no cover - allows importing the module (e.g. for docs) without mink
    Limit = object  # type: ignore[misc,assignment]
    Constraint = None  # type: ignore[assignment]

__all__ = ["DropbearMotorLimit", "pair_inverse_unclipped", "G1_SHOULDER_COMFORT", "comfort_limit"]


def pair_inverse_unclipped(pair, pr: np.ndarray, iters: int = 12) -> np.ndarray:
    """(pitch, roll) (..., 2) -> calf motors (a, b) (..., 2) WITHOUT clipping (linear extrapolation of the table)."""
    guess, _, _ = _bilinear(pair.pg, pair.rg, pair.inv_guess, pr[..., 0], pr[..., 1])
    a, b = guess[..., 0], guess[..., 1]
    for _ in range(iters):
        val, da, db = _bilinear(pair.ag, pair.bg, pair.table, a, b)
        err = pr - val
        j00, j10, j01, j11 = da[..., 0], da[..., 1], db[..., 0], db[..., 1]
        det = j00 * j11 - j01 * j10
        ok = np.abs(det) > 1e-9
        det = np.where(ok, det, 1.0)
        step_a = np.where(ok, (j11 * err[..., 0] - j01 * err[..., 1]) / det, 0.0)
        step_b = np.where(ok, (-j10 * err[..., 0] + j00 * err[..., 1]) / det, 0.0)
        shrink = np.maximum(1.0, np.hypot(step_a, step_b) / 0.1)
        a, b = a + step_a / shrink, b + step_b / shrink
    return np.stack([a, b], axis=-1)


class DropbearMotorLimit(Limit):
    """Linearised motor-limit inequalities for the serial3 groups and ankle pairs (see module doc).

    Args:
        model: MuJoCo model of the serial Dropbear MJCF (joints named like the semantic DOFs).
        smap: the calibration's SemanticMap.
        gain: fraction of the remaining motor margin allowed per IK step (like mink's ConfigurationLimit).
        margin: rad kept away from each motor limit.
        h: finite-difference step [rad].
    """

    def __init__(self, model, smap: SemanticMap, gain: float = 0.95, margin: float = 1e-3, h: float = 1e-4):
        import mujoco

        self.model = model
        self.smap = smap
        self.gain, self.margin, self.h = gain, margin, h
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SEMANTIC_NAMES]
        if min(jid) < 0:
            raise ValueError("model lacks semantic joints")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.lim = smap.motor_limits
        self.groups = []   # (semantic idx (k,), motor idx (k,), inverse fn)
        for g in smap._serial:
            self.groups.append((np.array(g.sem), np.array(g.chain), g.inv))
        for ip, ir, pair in smap._pairs:
            self.groups.append((np.array([ip, ir]), np.array([pair.ma, pair.mb]),
                                lambda s, p=pair: pair_inverse_unclipped(p, s)))
        self.stats = {"calls": 0, "active_rows_max": 0}

    def motor_values(self, s: np.ndarray) -> np.ndarray:
        """Unclipped motors of the constrained groups for a semantic vector (22,) -> dict-like (22,) with NaN elsewhere."""
        out = np.full(22, np.nan)
        for sem, mot, inv in self.groups:
            out[mot] = inv(s[sem][None])[0]
        return out

    def compute_qp_inequalities(self, configuration, dt: float):  # noqa: D401 - mink API
        del dt
        q = configuration.q
        s = q[self.qadr]
        rows_g, rows_h = [], []
        for sem, mot, inv in self.groups:
            k = len(sem)
            pts = np.repeat(s[sem][None], 1 + 2 * k, axis=0)
            for i in range(k):
                pts[1 + 2 * i, i] += self.h
                pts[2 + 2 * i, i] -= self.h
            m = inv(pts)                                   # (1+2k, k)
            m0 = m[0]
            jac = np.stack([(m[1 + 2 * i] - m[2 + 2 * i]) / (2 * self.h) for i in range(k)], axis=1)  # (k motors, k sem)
            lo = self.lim[mot, 0] + self.margin
            hi = self.lim[mot, 1] - self.margin
            g = np.zeros((2 * k, self.model.nv))
            g[:k, self.dadr[sem]] = jac
            g[k:, self.dadr[sem]] = -jac
            rows_g.append(g)
            rows_h.append(np.concatenate([self.gain * (hi - m0), self.gain * (m0 - lo)]))
        self.stats["calls"] += 1
        return Constraint(G=np.vstack(rows_g), h=np.concatenate(rows_h))


# G1 anatomical shoulder ranges [rad] from GMR assets/unitree_g1/g1_mocap_29dof.xml (joint names mapped to the
# dropbear-semantic-v1 names, which use G1 conventions). Dropbear's own shoulder ranges are the full +-180 deg of
# its shoulder motors, so the YXZ shoulder angles have two equivalent branches, (p, r, y) and (p+pi, pi-r, y+pi):
# with a noisy human upper-arm twist target the IK can settle on the flipped branch (same arm pose, shoulder motors
# rotated ~180 deg). These RETARGETING-level bounds (not model limits) keep the IK on the anatomical branch.
G1_SHOULDER_COMFORT: dict[str, tuple[float, float]] = {
    "left_shoulder_pitch": (-3.0892, 1.149), "left_shoulder_roll": (-0.6, 2.2515), "left_shoulder_yaw": (-1.4, 2.0),
    "right_shoulder_pitch": (-3.0892, 1.149), "right_shoulder_roll": (-2.2515, 0.6), "right_shoulder_yaw": (-2.0, 1.4),
}


def comfort_limit(model, ranges: dict[str, tuple[float, float]] | None = None, gain: float = 0.95):
    """A mink ``ConfigurationLimit`` on a copy of ``model`` whose listed joint ranges are intersected with ``ranges``."""
    import copy

    import mink
    import mujoco

    ranges = G1_SHOULDER_COMFORT if ranges is None else ranges
    m2 = copy.deepcopy(model)
    applied = {}
    for n, (lo, hi) in ranges.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0:
            raise KeyError(n)
        a, b = model.jnt_range[j]
        m2.jnt_range[j] = [max(a, lo), min(b, hi)]
        applied[n] = [float(m2.jnt_range[j][0]), float(m2.jnt_range[j][1])]
    lim = mink.ConfigurationLimit(m2, gain=gain)
    lim.applied_ranges = applied
    return lim
