"""Analytic arm seed for IK on the DERIVED serial Dropbear model (pure numpy).

Dropbear's shoulder is a full-range YXZ (pitch -> roll -> yaw) chain followed by the elbow hinge (axis ~ +y of the
upper arm, G1 convention: elbow 0 = forearm forward, +pi/2 = straight arm). With +-180 deg shoulder ranges the same
arm pose has two Euler representations, ``(p, r, y)`` and ``(p + pi, pi - r, y + pi)``, and a local IK started from
the previous frame can drift onto the "flipped" one or get stuck at a limit corner. :func:`arm_seed` computes the arm
joints that point the upper arm along ``u`` and the forearm along ``f`` (pelvis-frame unit vectors) exactly:

* elbow: ``e = asin(u . f)`` (forearm in the upper-arm frame is ``(cos e, 0, -sin e)``);
* upper-arm frame ``F``: ``F @ (0, 0, -1) = u`` and ``F @ (cos e, 0, -sin e) = f``;
* shoulder angles: the YXZ decomposition of ``F``. Branch choice: on the FIRST frame the representation inside the
  anatomical ``comfort`` ranges (G1's); afterwards CONTINUITY first -- each candidate is unwrapped (+-2 pi per
  angle, joints cannot wrap) to the value closest to the previous solution and the closest one inside the model
  ranges wins, so the joint trajectory never jumps between branches (a branch change happens only continuously,
  through the roll = +-90 deg singularity where both representations meet).

When the arm is (nearly) straight the elbow plane is undefined and the previous upper-arm x axis is kept.
"""
from __future__ import annotations

import numpy as np

from dropbear_wbc.kinematics.semantic import euler_yxz, euler_yxz_matrix

__all__ = ["arm_seed", "wrap", "arm_frame"]


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def arm_frame(u: np.ndarray, f: np.ndarray, prev: np.ndarray, straight_tol: float = 0.15):
    """(upper-arm frame F (3, 3), elbow e) for directions ``u`` (shoulder->elbow) and ``f`` (elbow->wrist)."""
    u = np.asarray(u, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u = u / max(np.linalg.norm(u), 1e-12)
    f = f / max(np.linalg.norm(f), 1e-12)
    s = float(np.clip(np.dot(u, f), -1.0, 1.0))
    e = float(np.arcsin(s))
    ce = float(np.cos(e))
    z = -u
    if ce > straight_tol:
        x = (f - s * u) / ce
    else:  # straight arm: keep the previous frame's upper-arm x axis (projected orthogonal to u)
        x = euler_yxz_matrix(np.asarray(prev[:3], dtype=np.float64))[:, 0]
    x = x - np.dot(x, z) * z
    n = np.linalg.norm(x)
    if n < 1e-9:  # degenerate: any axis orthogonal to u
        x = np.cross(z, [0.0, 1.0, 0.0])
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(z, [1.0, 0.0, 0.0])
        n = np.linalg.norm(x)
    x = x / n
    return np.stack([x, np.cross(z, x), z], axis=1), e


def arm_seed(u: np.ndarray, f: np.ndarray, prev: np.ndarray, ranges: np.ndarray, comfort: np.ndarray | None = None,
             first: bool = False, straight_tol: float = 0.15) -> np.ndarray:
    """Shoulder (pitch, roll, yaw) + elbow for upper-arm direction ``u`` and forearm direction ``f``.

    Args:
        u, f: (3,) directions in the pelvis frame (need not be unit).
        prev: (4,) previous (pitch, roll, yaw, elbow) [rad].
        ranges: (4, 2) model joint ranges [rad] (hard).
        comfort: (3, 2) anatomical shoulder ranges used for the first-frame branch choice (default: ``ranges``).
        first: first frame of a sequence (branch by comfort, not continuity).
    Returns (4,) seed [rad] (clipped to ``ranges``).
    """
    fm, e = arm_frame(u, f, prev, straight_tol)
    a = euler_yxz(fm)
    cands = [wrap(a), wrap(np.array([a[0] + np.pi, np.pi - a[1], a[2] + np.pi]))]
    lo, hi = ranges[:3, 0], ranges[:3, 1]
    pv = np.asarray(prev[:3], dtype=np.float64)
    best, best_cost = None, np.inf
    for c in cands:
        if first:
            clo, chi = (comfort[:, 0], comfort[:, 1]) if comfort is not None else (lo, hi)
            cost = np.sum(np.maximum(clo - c, 0.0) + np.maximum(c - chi, 0.0))
            cand = c
        else:
            cand = pv + wrap(c - pv)  # unwrapped to the nearest value of each angle
            for k in range(3):  # joints cannot wrap: bring back inside the model range if possible
                if cand[k] > hi[k]:
                    cand[k] -= 2 * np.pi
                elif cand[k] < lo[k]:
                    cand[k] += 2 * np.pi
            out = np.sum(np.maximum(lo - cand, 0.0) + np.maximum(cand - hi, 0.0))
            cost = 100.0 * out + np.sum(np.abs(cand - pv))
        if cost < best_cost:
            best, best_cost = cand, cost
    seed = np.concatenate([best, [e]])
    return np.clip(seed, ranges[:, 0], ranges[:, 1])
