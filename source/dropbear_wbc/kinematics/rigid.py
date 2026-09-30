"""Small vectorised rigid-body helpers (numpy) used by the calibration fit.

Conventions: rotation matrices act on column vectors (``v_parent = R @ v_child``); quaternions wxyz;
angles radians; lengths metres. Intrinsic Euler 'YXZ' means ``R = Ry(a) @ Rx(b) @ Rz(c)`` (G1 hip /
shoulder order pitch -> roll -> yaw).
"""
from __future__ import annotations

import numpy as np

from dropbear_wbc.motion.rotations import matrix_to_euler_intrinsic, quat_to_matrix


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Vectorised rotation matrix (..., 3, 3) -> wxyz quaternion (..., 4) with w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    shp = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    tr = np.trace(m, axis1=1, axis2=2)
    cand = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1)
    k = np.argmax(cand, axis=1)
    q = np.empty((m.shape[0], 4))
    # case w largest
    s = np.sqrt(np.maximum(1.0 + cand[:, 0], 1e-300)) * 2
    q0 = np.stack([0.25 * s, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s], 1)
    s = np.sqrt(np.maximum(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1e-300)) * 2
    q1 = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, 0.25 * s, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s], 1)
    s = np.sqrt(np.maximum(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], 1e-300)) * 2
    q2 = np.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, 0.25 * s, (m[:, 1, 2] + m[:, 2, 1]) / s], 1)
    s = np.sqrt(np.maximum(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], 1e-300)) * 2
    q3 = np.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s], 1)
    for i, qi in enumerate((q0, q1, q2, q3)):
        q[k == i] = qi[k == i]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(q[:, :1] < 0, -q, q)
    return q.reshape(shp + (4,))


def rotvec(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3) (axis * angle, angle in [0, pi])."""
    q = mat_to_quat(m)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 0])
    scale = np.where(s > 1e-12, ang / np.maximum(s, 1e-300), 2.0)
    return v * scale[..., None]


def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues: unit ``axis`` (..., 3), ``angle`` (...) -> (..., 3, 3)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.asarray(angle, dtype=np.float64)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    m = np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y,
                  t * x * y + s * z, t * y * y + c, t * y * z - s * x,
                  t * x * z - s * y, t * y * z + s * x, t * z * z + c], axis=-1)
    return m.reshape(np.broadcast_shapes(axis.shape[:-1], angle.shape) + (3, 3))


def signed_angle_about(m: np.ndarray, ref_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation angle of ``m`` (..., 3, 3) signed by its axis' direction w.r.t. ``ref_axis`` (3,).

    Returns (signed angle (...), unit rotation axis flipped to have a non-negative dot with ref (..., 3)).
    """
    rv = rotvec(m)
    ang = np.linalg.norm(rv, axis=-1)
    ax = np.where(ang[..., None] > 1e-12, rv / np.maximum(ang[..., None], 1e-300), ref_axis)
    sgn = np.where(np.sum(ax * ref_axis, axis=-1) >= 0, 1.0, -1.0)
    return sgn * ang, ax * sgn[..., None]


def euler_yxz(m: np.ndarray) -> np.ndarray:
    """``R = Ry(a) Rx(b) Rz(c)`` -> (a, b, c) = (pitch, roll, yaw) (..., 3)."""
    return matrix_to_euler_intrinsic("YXZ", m)


def quat_mat(q: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q)


def fit_screw_axis(r_rel: np.ndarray, p_rel: np.ndarray, angles: np.ndarray) -> dict:
    """Fit a fixed revolute axis to relative motions ``T_i = (R_i, p_i)`` of a body at joint ``angles``.

    Returns dict with ``axis`` (unit, sign such that the rotation angle increases with ``angles``),
    ``point`` (least-squares point on the axis, closest to the origin), ``slope`` (rotation angle per
    joint unit, least squares), ``rot_rms`` (rad, residual of R_i vs Rot(axis, slope * angle_i)).
    """
    rv = rotvec(r_rel)
    # principal direction of rotation vectors
    u, s, vt = np.linalg.svd(rv - 0.0, full_matrices=False)
    axis = vt[0]
    proj = rv @ axis
    if np.polyfit(angles, proj, 1)[0] < 0:
        axis, proj = -axis, -proj
    slope = float(np.dot(angles, proj) / max(np.dot(angles, angles), 1e-300))
    model = axis_angle_matrix(np.broadcast_to(axis, rv.shape), slope * angles)
    err = rotvec(np.swapaxes(model, -1, -2) @ r_rel)
    rot_rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=-1))))
    # point: (I - R_i) x = p_i  for all i (least squares), minimum-norm along the axis
    a = (np.eye(3)[None] - r_rel).reshape(-1, 3)
    b = p_rel.reshape(-1)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    x = x - axis * np.dot(x, axis)
    resid = (np.eye(3)[None] - r_rel) @ x - p_rel
    return {"axis": axis, "point": x, "slope": slope, "rot_rms": rot_rms,
            "trans_rms": float(np.sqrt(np.mean(np.sum(resid ** 2, axis=-1))))}
