"""Gravity feed-forward for the Dropbear arm motors (xr_teleoperate ``pin.rnea(model, q, 0, 0)`` equivalent).

Model: the arm link masses and centres of mass extracted from the Newton plant (``tools/extract_arm_inertia.py``
-> ``data/teleop/arm_inertia.json``) are attached to the semantic serial chain of :mod:`.arm_ik` (each body to the
segment it moves with). The holding torque of semantic joint j is

    tau_sem_j = sum_b m_b g (w_j x (c_b - p_j)) . z        (bodies b beyond joint j)

(``+dU/dq`` of the potential ``U = sum m g z``), and the motor torque follows by virtual work through the measured
semantic map, ``tau_motor = (d q_sem / d q_motor)^T tau_sem``. For the elbow four-bar the motor side is exact
(1-DoF mechanism: ``tau_m = tau_e * de/dm``, the local slope of the elbow table, about -4); the four-bar links
themselves (about 0.11 kg per arm) are lumped with the upper arm.

Forearm path (review fix 2026-09-24): the elbow is polycentric. With ``forearm_model="table"`` (default whenever the
IK chain carries the measured wrist path, i.e. the IK's own default ``elbow_model="table"``), the forearm and hand
translate with the MEASURED wrist point ``W(e)`` (CalibrationFK) while rotating by the semantic elbow angle, and
``tau_e = dU/de`` is taken numerically; the shoulder torques use the same corrected COM positions. ``"pivot"`` is the
previous rigid best-fit pivot (4.5 mm p50 / 14 mm max off the measured path, ``fk_vs_calibration.log``).
Validation (CPU Newton, passive damping 0): ``logs/review_fixes/probe_shoulder_pd0_analysis.*``,
``logs/review_fixes/gravity_forearm_model_compare.log``.

Usage::

    grav = ArmGravity(ik)                          # ik: DropbearArmIK
    tau10 = grav.motor_torque(q_sem10, motor10)    # feed-forward for SDK slots 12..21 [N*m]
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .arm_ik import _AXIS_IDS, _rot, ARM_MOTOR_SLOTS, SIDES, ArmChain, DropbearArmIK

DEFAULT_INERTIA = Path(__file__).resolve().parents[3] / "data" / "teleop" / "arm_inertia.json"
GRAVITY = 9.81


def _partials(chain: ArmChain, q: np.ndarray):
    """World axis w_j and point p_j of every joint, and the rigid transform (R_k, t_k) after joints 1..k."""
    r = np.eye(3)
    t = np.zeros(3)
    ws, ps, tfs = [], [], [(r.copy(), t.copy())]
    for i in range(5):
        a = _AXIS_IDS[i]
        ws.append(r[:, a].copy())
        ps.append(r @ chain.points[i] + t)
        ri = _rot(a, float(q[i]))
        t = t + r @ (chain.points[i] - ri @ chain.points[i])
        r = r @ ri
        tfs.append((r.copy(), t.copy()))
    return ws, ps, tfs


class ArmGravity:
    """Arm gravity torques from the extracted inertials on the semantic chain (see module docstring)."""

    def __init__(self, ik: DropbearArmIK, inertia: str | Path = DEFAULT_INERTIA, g: float = GRAVITY,
                 forearm_model: str = "table"):
        if forearm_model not in ("table", "pivot"):
            raise ValueError(f"forearm_model must be 'table' or 'pivot', got {forearm_model!r}")
        self.forearm_model = forearm_model
        self.ik = ik
        self.path = Path(inertia)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("schema") != "dropbear-teleop-arm-inertia-v1":
            raise ValueError(f"{self.path}: unexpected schema {self.data.get('schema')!r}")
        self.g = float(g)
        mq = np.asarray(self.data["motor_q_at_extraction"], dtype=float)
        if np.abs(mq[list(ARM_MOTOR_SLOTS)]).max() > 1e-6:
            raise ValueError("arm_inertia.json must be extracted at the authored arm configuration (arm motors 0)")
        self.bodies: dict[str, list[tuple[int, float, np.ndarray]]] = {s: [] for s in SIDES}
        self.lumped: dict[str, list[tuple[int, float, np.ndarray]]] = {s: [] for s in SIDES}
        self._build()

    def _build(self) -> None:
        for side in SIDES:
            ch = self.ik.chains[side]
            # the authored arm: shoulder 0, straight elbow (semantic e_rest), wrist roll 0
            q_ext = np.array([0.0, 0.0, 0.0, ch.info["elbow_rest_rad"], 0.0])
            _, _, tfs = _partials(ch, q_ext)
            self.bodies[side] = []
            for name, b in self.data["bodies"].items():
                if b["side"] != side:
                    continue
                k = int(b["segment"])
                r, t = tfs[k]
                c0 = r.T @ (np.asarray(b["com_root"], dtype=float) - t)  # COM at the semantic zero configuration
                self.bodies[side].append((k, float(b["mass_kg"]), c0))
            # gravity is linear in the first mass moments: lump the bodies of each segment (exact)
            self.lumped[side] = []
            for k in range(1, 6):
                mk = sum(m for kk, m, _ in self.bodies[side] if kk == k)
                if mk > 0:
                    sk = sum(m * c0 for kk, m, c0 in self.bodies[side] if kk == k)
                    self.lumped[side].append((k, mk, sk))

    def reload(self) -> None:
        """Rebuild after the IK reloaded its calibration (the chain geometry may have moved)."""
        self._build()

    @property
    def arm_mass(self) -> dict[str, float]:
        return {s: sum(m for _, m, _ in self.bodies[s]) for s in SIDES}

    def _forearm_offset(self, ch: ArmChain, q5: np.ndarray, tfs) -> np.ndarray:
        """World translation that moves the rigid-pivot forearm/hand onto the measured wrist path (zero for 'pivot')."""
        if self.forearm_model != "table" or ch.elbow_table is None:
            return np.zeros(3)
        r_sh, t_sh = tfs[3]
        r4, t4 = tfs[4]  # after the elbow, before the wrist roll: the shift must not depend on the wrist roll
        w_e, _ = ch._wrist_zero(float(q5[3]))
        return (r_sh @ w_e + t_sh) - (r4 @ ch.wrist0 + t4)

    def _forearm_potential(self, side: str, q5: np.ndarray) -> float:
        """U/g of the bodies beyond the elbow (z first moment) at semantic q (5,)."""
        ch = self.ik.chains[side]
        _, _, tfs = _partials(ch, q5)
        off = self._forearm_offset(ch, q5, tfs)
        u = 0.0
        for k, m, s in self.lumped[side]:
            if k >= 4:
                r, t = tfs[k]
                u += float((r @ s + m * (t + off))[2])
        return u

    def uses_table(self, side: str) -> bool:
        return self.forearm_model == "table" and self.ik.chains[side].elbow_table is not None

    def semantic_torque(self, side: str, q5: np.ndarray) -> np.ndarray:
        """Holding torque (5,) [N*m] of the semantic joints of one arm at semantic q (5,)."""
        ch = self.ik.chains[side]
        q5 = np.asarray(q5, dtype=float)
        ws, ps, tfs = _partials(ch, q5)
        off = self._forearm_offset(ch, q5, tfs)
        tau = np.zeros(5)
        # accumulate, from the tip, the mass M_j and first moment S_j (world) of everything beyond joint j
        m_acc, s_acc = 0.0, np.zeros(3)
        lumped = {k: (m, s) for k, m, s in self.lumped[side]}
        for j in range(4, -1, -1):
            if j + 1 in lumped:
                m, s = lumped[j + 1]
                r, t = tfs[j + 1]
                m_acc += m
                s_acc = s_acc + r @ s + m * (t + (off if j + 1 >= 4 else 0.0))
            if m_acc > 0.0:
                w, p = ws[j], ps[j] + (off if j >= 4 else 0.0)  # the wrist-roll axis moves with the forearm
                v = s_acc - m_acc * p
                tau[j] = self.g * (w[0] * v[1] - w[1] * v[0])  # (w x v) . z
        if self.uses_table(side):  # polycentric elbow: tau_e = dU/de along the measured path
            h = 1e-4
            qp, qm = q5.copy(), q5.copy()
            qp[3] += h
            qm[3] -= h
            tau[3] = self.g * (self._forearm_potential(side, qp) - self._forearm_potential(side, qm)) / (2 * h)
        return tau

    def semantic_torque_both(self, q_sem10: np.ndarray) -> np.ndarray:
        q = np.asarray(q_sem10, dtype=float)
        return np.concatenate([self.semantic_torque("left", q[:5]), self.semantic_torque("right", q[5:])])

    def sem_motor_jacobian(self, motor10: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """d q_sem(arms, 10) / d q_motor(arms, 10) by central differences of ``SemanticMap.motor_to_semantic``
        (one batched call), one-sided at motor limits / table ends."""
        base = self.ik.motor22(np.asarray(motor10, dtype=float))
        lim = self.ik.smap.motor_limits
        slots = list(ARM_MOTOR_SLOTS)
        batch = [base]
        steps = []
        for i in slots:
            hp = min(h, lim[i, 1] - base[i])
            hm = min(h, base[i] - lim[i, 0])
            qp, qm = base.copy(), base.copy()
            qp[i] += hp
            qm[i] -= hm
            batch += [qp, qm]
            steps.append(hp + hm)
        sem = self.ik.motor_to_semantic_arms_fast(np.stack(batch))
        jac = np.zeros((10, 10))
        for n, st in enumerate(steps):
            if st > 0:
                jac[:, n] = (sem[1 + 2 * n] - sem[2 + 2 * n]) / st
        return jac

    def motor_torque(self, q_sem10: np.ndarray, motor10: np.ndarray | None = None) -> np.ndarray:
        """Gravity feed-forward (10,) [N*m] for the arm motors (SDK slots 12..21) at semantic ``q_sem10``.

        ``motor10`` (the matching motor angles) is used for the map derivative; default ``ik.to_motor_fast``."""
        if motor10 is None:
            motor10, _ = self.ik.to_motor_fast(q_sem10)
        tau_sem = self.semantic_torque_both(q_sem10)
        return self.sem_motor_jacobian(motor10).T @ tau_sem
