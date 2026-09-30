"""Pose composition and finite-difference velocities for the motion NPZ.

Velocity conventions follow BeyondMimic ``scripts/csv_to_npz.py``:

* linear quantities: ``np.gradient(x, dt, axis=0)`` (central differences, one-sided at the ends);
* angular velocity: ``omega[t] = axis_angle(q[t+1] * q[t-1]^-1) / (2 dt)`` in the world frame, first and
  last samples repeated.

``body_lin_vel_w`` is the velocity of each body's **centre of mass** (Isaac Lab 2.2
``body_lin_vel_w`` == ``body_com_lin_vel_w``; the tracking reward compares against it), obtained by
differentiating COM positions ``p_link + R_link @ com_b``. ``body_ang_vel_w`` is frame independent.
Units: m, rad, s. Quaternions wxyz.
"""
from __future__ import annotations

import numpy as np

from dropbear_wbc.motion.rotations import quat_conj, quat_mul, quat_to_matrix


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion(s) -> rotation vector(s) (..., 3), angle in [0, pi]."""
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    scale = np.where(s > 1e-9, angle / np.maximum(s, 1e-12), 2.0)
    return v * scale[..., None]


def compose_root(root_pos: np.ndarray, root_quat: np.ndarray, rel_pos: np.ndarray, rel_quat: np.ndarray):
    """World body poses from root poses (T,3)/(T,4) and root-relative body poses (T,B,3)/(T,B,4)."""
    r = quat_to_matrix(root_quat)  # (T,3,3)
    pos = root_pos[:, None, :] + np.einsum("tij,tbj->tbi", r, rel_pos)
    quat = quat_mul(np.broadcast_to(root_quat[:, None, :], rel_quat.shape), rel_quat)
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    return pos, quat


def angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity (T, ..., 3) from orientations (T, ..., 4) by central SO(3) differences."""
    t = quat.shape[0]
    if t < 3:
        return np.zeros(quat.shape[:-1] + (3,))
    q_rel = quat_mul(quat[2:], quat_conj(quat[:-2]))
    omega = quat_to_axis_angle(q_rel) / (2.0 * dt)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


def linear_velocity(x: np.ndarray, dt: float) -> np.ndarray:
    """``np.gradient`` along time (axis 0); zeros for a single frame."""
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0)


def com_positions(body_pos: np.ndarray, body_quat: np.ndarray, com_b: np.ndarray) -> np.ndarray:
    """COM world positions (T,B,3) from link poses and link-frame COM offsets ``com_b`` (B,3)."""
    return body_pos + np.einsum("tbij,bj->tbi", quat_to_matrix(body_quat), com_b)


def unwrap_joints(joint_pos: np.ndarray, revolute_mask: np.ndarray) -> np.ndarray:
    """Unwrap revolute joint trajectories (T,J) across +-pi jumps (prismatic columns untouched)."""
    out = joint_pos.copy()
    if out.shape[0] > 1 and revolute_mask.any():
        out[:, revolute_mask] = np.unwrap(out[:, revolute_mask], axis=0)
    return out
