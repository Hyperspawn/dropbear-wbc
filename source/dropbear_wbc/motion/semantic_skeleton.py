"""Approximate Dropbear leg kinematics in *semantic* joint space (legs only, numpy).

This is NOT the Dropbear plant (the plant has four-bar knees and a parallel ankle and lives in the
USD). It is an idealised serial leg with the G1 semantic axis order, built from calibration segment
lengths, used for:

* per-frame root-height correction so stance feet touch z = 0 after retargeting,
* foot-slip / penetration metrics of retargeted clips,
* inverse kinematics of the synthetic clips (squat, weight shift).

Chain per side (pelvis frame: x fwd, y left, z up; right side mirrors y)::

    hip   = (0, +-w/2, 0)
    R_th  = Ry(hip_pitch) Rx(hip_roll) Rz(hip_yaw)
    knee  = hip  + R_th @ (0, 0, -L_thigh)
    R_sh  = R_th Ry(knee)
    ankle = knee + R_sh @ (a_fwd, 0, -L_shank)
    R_ft  = R_sh Ry(ankle_pitch) Rx(ankle_roll)
    soles = ankle + R_ft @ {(+front, +-fw, -h), (-back, +-fw, -h)}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .names import SEMANTIC_INDEX
from .rotations import rot_x, rot_y, rot_z

__all__ = ["LegGeometry", "SemanticLegFK", "leg_fk", "planar_leg_ik", "leg_ik"]

LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


@dataclass(frozen=True)
class LegGeometry:
    hip_width: float
    thigh: float
    shank: float
    ankle_height: float
    ankle_forward: float = 0.0
    foot_front: float = 0.18
    foot_back: float = 0.11
    foot_half_width: float = 0.04

    @property
    def straight_hip_height(self) -> float:
        return self.thigh + self.shank + self.ankle_height

    @classmethod
    def from_calibration(cls, cal: object) -> "LegGeometry":
        return cls(
            hip_width=float(getattr(cal, "hip_width")),
            thigh=float(getattr(cal, "hip_to_knee")),
            shank=float(getattr(cal, "knee_to_ankle")),
            ankle_height=float(getattr(cal, "ankle_to_sole")),
            ankle_forward=float(getattr(cal, "ankle_forward_of_hip")),
            foot_front=float(getattr(cal, "foot_front")),
            foot_back=float(getattr(cal, "foot_back")),
            foot_half_width=float(getattr(cal, "foot_half_width", 0.04)),
        )


@dataclass
class SemanticLegFK:
    """World-frame results for T frames: ``sole`` (T, 2, 4, 3), ``ankle``/``knee``/``hip`` (T, 2, 3)."""

    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    sole: np.ndarray

    @property
    def sole_height(self) -> np.ndarray:
        """(T, 2) lowest sole point z per foot."""
        return self.sole[..., 2].min(axis=-1)

    @property
    def foot_center(self) -> np.ndarray:
        """(T, 2, 3) mean sole point per foot."""
        return self.sole.mean(axis=2)


def _leg_angles(q_sem: np.ndarray, side: str) -> dict[str, np.ndarray]:
    return {j: q_sem[..., SEMANTIC_INDEX[f"{side}_{j}"]] for j in LEG_JOINTS}


def leg_fk(
    pelvis_pos: np.ndarray, pelvis_rot: np.ndarray, q_sem: np.ndarray, geom: LegGeometry
) -> SemanticLegFK:
    """Semantic-space leg FK. ``pelvis_pos`` (T,3), ``pelvis_rot`` (T,3,3), ``q_sem`` (T,22) [rad]."""
    t = q_sem.shape[0]
    hips, knees, ankles = np.empty((t, 2, 3)), np.empty((t, 2, 3)), np.empty((t, 2, 3))
    soles = np.empty((t, 2, 4, 3))
    for s, side in enumerate(("left", "right")):
        sgn = 1.0 if side == "left" else -1.0
        a = _leg_angles(q_sem, side)
        hip = pelvis_pos + pelvis_rot @ np.array([0.0, sgn * geom.hip_width / 2, 0.0])
        r_th = pelvis_rot @ rot_y(a["hip_pitch"]) @ rot_x(a["hip_roll"]) @ rot_z(a["hip_yaw"])
        knee = hip + r_th @ np.array([0.0, 0.0, -geom.thigh])
        r_sh = r_th @ rot_y(a["knee"])
        ankle = knee + r_sh @ np.array([geom.ankle_forward, 0.0, -geom.shank])
        r_ft = r_sh @ rot_y(a["ankle_pitch"]) @ rot_x(a["ankle_roll"])
        pts = np.array(
            [
                [geom.foot_front, geom.foot_half_width, -geom.ankle_height],
                [geom.foot_front, -geom.foot_half_width, -geom.ankle_height],
                [-geom.foot_back, geom.foot_half_width, -geom.ankle_height],
                [-geom.foot_back, -geom.foot_half_width, -geom.ankle_height],
            ]
        )
        hips[:, s], knees[:, s], ankles[:, s] = hip, knee, ankle
        soles[:, s] = ankle[:, None, :] + np.einsum("tij,kj->tki", r_ft, pts)
    return SemanticLegFK(hip=hips, knee=knees, ankle=ankles, sole=soles)


def planar_leg_ik(
    hip_to_ankle_x: np.ndarray, hip_to_ankle_z: np.ndarray, geom: LegGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sagittal 2-link IK with the foot kept flat (pelvis upright).

    Inputs: ankle position relative to the hip in the pelvis x/z plane [m] (z negative = below).
    Returns ``(hip_pitch, knee, ankle_pitch)`` [rad], knee >= 0 (G1 flexion positive), such that
    ``hip_pitch + knee + ankle_pitch = 0`` (flat foot). Raises if the target is unreachable.
    """
    x = np.asarray(hip_to_ankle_x, dtype=np.float64)
    z = np.asarray(hip_to_ankle_z, dtype=np.float64)
    # The shank vector (a_fwd, 0, -L) in the shank frame equals an effective straight shank of length
    # L' = hypot(a_fwd, L) whose knee angle is offset by gamma = atan2(a_fwd, L):
    #   Ry(k) @ (a, 0, -L) = L' * (-sin(k - gamma), 0, -cos(k - gamma)).
    l1 = geom.thigh
    l2 = float(np.hypot(geom.ankle_forward, geom.shank))
    gamma = float(np.arctan2(geom.ankle_forward, geom.shank))
    d2 = x * x + z * z
    c = (d2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if np.any(c > 1.0 + 1e-9) or np.any(c < -1.0):
        raise ValueError("planar_leg_ik: target out of reach")
    k_eff = np.arccos(np.clip(c, -1.0, 1.0))
    # Thigh direction: Ry(hp) @ (0,0,-1) = (-sin hp, 0, -cos hp); the effective shank adds k_eff:
    # ankle = l1*(-sin hp, -cos hp) + l2*(-sin(hp+k_eff), -cos(hp+k_eff))  in (x, z).
    phi = np.arctan2(-x, -z)
    beta = np.arctan2(l2 * np.sin(k_eff), l1 + l2 * np.cos(k_eff))
    hip_pitch = phi - beta
    knee = k_eff + gamma
    ankle_pitch = -(hip_pitch + knee)
    return hip_pitch, knee, ankle_pitch


_IK_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


def _single_leg(
    pelvis_pos: np.ndarray, pelvis_rot: np.ndarray, ang: dict[str, np.ndarray], side: str, geom: LegGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Ankle position (T,3) and foot rotation (T,3,3) of one leg in world."""
    sgn = 1.0 if side == "left" else -1.0
    hip = pelvis_pos + pelvis_rot @ np.array([0.0, sgn * geom.hip_width / 2, 0.0])
    r_th = pelvis_rot @ rot_y(ang["hip_pitch"]) @ rot_x(ang["hip_roll"]) @ rot_z(ang["hip_yaw"])
    knee = hip + r_th @ np.array([0.0, 0.0, -geom.thigh])
    r_sh = r_th @ rot_y(ang["knee"])
    ankle = knee + r_sh @ np.array([geom.ankle_forward, 0.0, -geom.shank])
    r_ft = r_sh @ rot_y(ang["ankle_pitch"]) @ rot_x(ang["ankle_roll"])
    return ankle, r_ft


def _rot_err(r_target: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    """Small-angle rotation error vector (T,3) of r_cur relative to r_target, in world."""
    e = r_cur @ np.swapaxes(r_target, -1, -2)
    return 0.5 * np.stack([e[..., 2, 1] - e[..., 1, 2], e[..., 0, 2] - e[..., 2, 0], e[..., 1, 0] - e[..., 0, 1]], -1)


def leg_ik(
    pelvis_pos: np.ndarray,
    pelvis_rot: np.ndarray,
    ankle_target: np.ndarray,
    foot_rot_target: np.ndarray,
    q_init: np.ndarray,
    geom: LegGeometry,
    iters: int = 30,
    damping: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Batched damped-least-squares IK on the semantic leg model (6 leg DoFs per side, square system).

    ``ankle_target`` (T,2,3) and ``foot_rot_target`` (T,2,3,3) in world. Solves hip pitch/roll/yaw, knee,
    ankle pitch/roll of both legs. Returns ``(q (T,22), max residual [m or rad])``.
    """
    q = np.array(q_init, dtype=np.float64, copy=True)
    eps = 1e-6
    worst = 0.0
    for s, side in enumerate(("left", "right")):
        cols = [SEMANTIC_INDEX[f"{side}_{j}"] for j in _IK_JOINTS]

        def residual(qq: np.ndarray) -> np.ndarray:
            ankle, r_ft = _single_leg(pelvis_pos, pelvis_rot, _leg_angles(qq, side), side, geom)
            return np.concatenate([ankle - ankle_target[:, s], _rot_err(foot_rot_target[:, s], r_ft)], axis=-1)

        for _ in range(iters):
            r = residual(q)  # (T, 6)
            jac = np.empty(r.shape + (len(cols),))
            for k, c in enumerate(cols):
                qp = q.copy()
                qp[:, c] += eps
                jac[..., k] = (residual(qp) - r) / eps
            jt = np.swapaxes(jac, -1, -2)
            h = jt @ jac + damping * np.eye(len(cols))
            dq = -np.linalg.solve(h, (jt @ r[..., None]))[..., 0]
            q[:, cols] += dq
            if np.abs(dq).max() < 1e-10:
                break
        worst = max(worst, float(np.abs(residual(q)).max()))
    return q, worst
