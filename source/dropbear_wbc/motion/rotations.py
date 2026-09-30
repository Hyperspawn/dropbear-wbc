"""Vectorised rotation helpers (pure numpy, no scipy) used by the motion pipeline.

Conventions
-----------
* Quaternions are **wxyz** unless a function name says ``xyzw``. They are
  Hamilton quaternions and ``q`` rotates vectors from the body frame into the
  parent/world frame: ``v_world = R(q) @ v_body``.
* Rotation matrices act on column vectors, so ``R_world_child = R_world_parent @ R_parent_child``.
* Angles are radians. Every function accepts arbitrary leading batch dimensions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_normalize",
    "quat_mul",
    "quat_conj",
    "quat_to_matrix",
    "matrix_to_quat",
    "quat_from_axis_angle",
    "quat_rotate",
    "quat_canonical",
    "quat_continuous",
    "quat_slerp",
    "xyzw_to_wxyz",
    "wxyz_to_xyzw",
    "rot_x",
    "rot_y",
    "rot_z",
    "euler_xyz_extrinsic_to_quat",
    "matrix_to_euler_intrinsic",
    "euler_intrinsic_to_matrix",
    "yaw_of_matrix",
    "rotation_angle",
    "rotation_log",
    "fit_euler_intrinsic",
]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q / |q|`` (wxyz or xyzw alike)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-norm quaternion")
    return q / n


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (wxyz)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion(s) -> rotation matrix/matrices (..., 3, 3)."""
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix/matrices (..., 3, 3) -> wxyz quaternion(s) with w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(m.shape[0]):
        r = m[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = quat_canonical(quat_normalize(q))
    return q.reshape(batch + (4,))


def quat_canonical(q: np.ndarray) -> np.ndarray:
    """Flip sign so that w >= 0 (same rotation)."""
    q = np.asarray(q, dtype=np.float64)
    return np.where(q[..., :1] < 0, -q, q)


def quat_continuous(q: np.ndarray) -> np.ndarray:
    """Remove sign flips along axis 0 of a (T, 4) quaternion sequence (for interpolation/CSV)."""
    q = np.array(q, dtype=np.float64, copy=True)
    for t in range(1, q.shape[0]):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Unit ``axis`` (..., 3) and ``angle`` (...) [rad] -> wxyz quaternion (..., 4)."""
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = 0.5 * angle
    return np.concatenate([np.cos(half)[..., None], np.sin(half)[..., None] * axis], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by quaternions ``q`` (..., 4, wxyz)."""
    return np.einsum("...ij,...j->...i", quat_to_matrix(q), np.asarray(v, dtype=np.float64))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical interpolation between wxyz quaternions, ``t`` in [0, 1] (broadcast over batch)."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_theta))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_theta))
    return quat_normalize(w0 * q0 + w1 * q1)


def rot_x(a: np.ndarray) -> np.ndarray:
    """Rotation matrices about +x by angle(s) ``a`` [rad] -> (..., 3, 3)."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([o, z, z, z, c, -s, z, s, c], axis=-1).reshape(a.shape + (3, 3))


def rot_y(a: np.ndarray) -> np.ndarray:
    """Rotation matrices about +y by angle(s) ``a`` [rad] -> (..., 3, 3)."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(a.shape + (3, 3))


def rot_z(a: np.ndarray) -> np.ndarray:
    """Rotation matrices about +z by angle(s) ``a`` [rad] -> (..., 3, 3)."""
    a = np.asarray(a, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([c, -s, z, s, c, z, z, z, o], axis=-1).reshape(a.shape + (3, 3))


_AXIS_FN = {"X": rot_x, "Y": rot_y, "Z": rot_z}


def euler_xyz_extrinsic_to_quat(angles: np.ndarray) -> np.ndarray:
    """Extrinsic x-y-z Euler angles (..., 3) [rad] -> wxyz quaternion.

    ``R = Rz(c) @ Ry(b) @ Rx(a)``. This equals ``scipy.Rotation.from_euler('xyz', ...)``
    (lower-case = extrinsic) and Warp's ``wp.quat_rpy(roll, pitch, yaw)``, which is the
    convention of soma-retargeter / BONES-SEED G1 CSVs.
    """
    angles = np.asarray(angles, dtype=np.float64)
    m = rot_z(angles[..., 2]) @ rot_y(angles[..., 1]) @ rot_x(angles[..., 0])
    return matrix_to_quat(m)


def euler_intrinsic_to_matrix(order: str, angles: np.ndarray) -> np.ndarray:
    """Intrinsic Euler (upper-case order, e.g. 'YXZ') -> ``R = R_a0 @ R_a1 @ R_a2``."""
    angles = np.asarray(angles, dtype=np.float64)
    m = _AXIS_FN[order[0]](angles[..., 0])
    for k in (1, 2):
        m = m @ _AXIS_FN[order[k]](angles[..., k])
    return m


def matrix_to_euler_intrinsic(order: str, m: np.ndarray) -> np.ndarray:
    """Decompose ``R = R_a0(t0) @ R_a1(t1) @ R_a2(t2)`` for Tait-Bryan orders (e.g. 'YXZ').

    Returns angles (..., 3) with the middle angle in [-pi/2, pi/2]. Near gimbal lock the
    split between the first and last angle is arbitrary (last angle set to 0).
    """
    m = np.asarray(m, dtype=np.float64)
    order = order.upper()
    if len(order) != 3 or len(set(order)) != 3:
        raise ValueError(f"only Tait-Bryan orders supported, got {order}")
    idx = {"X": 0, "Y": 1, "Z": 2}
    i, j, k = (idx[c] for c in order)
    # Parity sign of the permutation (i, j, k).
    sign = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    # For R = Ri(a) Rj(b) Rk(c):  R[i, k] = sign * sin(b)
    sb = np.clip(sign * m[..., i, k], -1.0, 1.0)
    b = np.arcsin(sb)
    cb = np.cos(b)
    gimbal = cb < 1e-7
    a = np.arctan2(-sign * m[..., j, k], m[..., k, k])
    c = np.arctan2(-sign * m[..., i, j], m[..., i, i])
    # Gimbal fallback: set c = 0 and recover a from the remaining block.
    a_g = np.arctan2(sign * m[..., k, j], m[..., j, j])
    a = np.where(gimbal, a_g, a)
    c = np.where(gimbal, 0.0, c)
    return np.stack([a, b, c], axis=-1)


def yaw_of_matrix(m: np.ndarray) -> np.ndarray:
    """Heading angle [rad] of the body x-axis projected on the world xy-plane."""
    m = np.asarray(m, dtype=np.float64)
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


def rotation_angle(m: np.ndarray) -> np.ndarray:
    """Geodesic angle [rad] of rotation matrix/matrices."""
    m = np.asarray(m, dtype=np.float64)
    tr = np.trace(m, axis1=-2, axis2=-1)
    return np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))


def rotation_log(m: np.ndarray) -> np.ndarray:
    """Rotation vector (..., 3) [rad] of rotation matrices (angle in [0, pi])."""
    m = np.asarray(m, dtype=np.float64)
    ang = rotation_angle(m)
    v = np.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
    s = np.sin(ang)
    small = s < 1e-6
    scale = np.where(small, 0.5, ang / (2.0 * np.where(small, 1.0, s)))
    out = v * scale[..., None]
    # Near pi the antisymmetric part vanishes; recover the axis from the symmetric part.
    near_pi = small & (ang > 3.0)
    if np.any(near_pi):
        mm = m[near_pi]
        diag = np.clip((np.diagonal(mm, axis1=-2, axis2=-1) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(diag)
        k = np.argmax(axis, axis=-1)
        for i in range(len(mm)):
            a = axis[i]
            for j in range(3):
                if j != k[i]:
                    a[j] = np.copysign(a[j], mm[i, k[i], j] + mm[i, j, k[i]])
            axis[i] = a / np.linalg.norm(a)
        out[near_pi] = axis * np.pi
    return out


def fit_euler_intrinsic(
    order: str,
    target: np.ndarray,
    seed: np.ndarray,
    iters: int = 30,
    prior: float = 1e-9,
    singular_prior: float = 0.0,
    singular_width: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler angles (..., 3) whose intrinsic ``order`` matrix equals ``target`` (..., 3, 3), found by
    damped Gauss-Newton started at ``seed`` (..., 3).

    Unlike the analytic decomposition this stays on the branch (and 2*pi wrap) of the seed and behaves
    continuously near gimbal lock: the Jacobian is analytic (spatial angular-velocity columns
    ``e_A, R_A e_B, R_A R_B e_C`` mapped by ``target^T``), so steps have no component along the
    unobservable direction, and a tiny ``prior`` pulls that direction towards the seed.

    ``singular_prior`` adds ``singular_prior * exp(-(cos(middle angle) / singular_width)^2)`` to the
    prior: negligible away from gimbal lock, strong near it, where it keeps the first/third angles near
    the seed instead of whirling (at the cost of a small orientation residual, which is returned).
    Returns ``(angles, residual_angle [rad])``.
    """
    order = order.upper()
    unit = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
    x = np.array(seed, dtype=np.float64, copy=True)
    seed = np.asarray(seed, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    tt = np.swapaxes(target, -1, -2)
    eye = np.eye(3)
    for _ in range(iters):
        ra = _AXIS_FN[order[0]](x[..., 0])
        rb = _AXIS_FN[order[1]](x[..., 1])
        rc = _AXIS_FN[order[2]](x[..., 2])
        r = rotation_log(tt @ (ra @ rb @ rc))
        c0 = np.broadcast_to(unit[order[0]], r.shape)
        c1 = ra @ unit[order[1]]
        c2 = ra @ rb @ unit[order[2]]
        jac = tt @ np.stack([c0, c1, c2], axis=-1)
        jt = np.swapaxes(jac, -1, -2)
        lam = prior + singular_prior * np.exp(-((np.cos(x[..., 1]) / singular_width) ** 2))
        h = jt @ jac + lam[..., None, None] * eye
        g = (jt @ r[..., None])[..., 0] + lam[..., None] * (x - seed)
        dx = -np.linalg.solve(h, g[..., None])[..., 0]
        n = np.linalg.norm(dx, axis=-1, keepdims=True)
        dx = np.where(n > 0.5, dx * (0.5 / np.maximum(n, 1e-12)), dx)
        x = x + dx
        if np.abs(dx).max() < 1e-12:
            break
    res = rotation_angle(tt @ euler_intrinsic_to_matrix(order, x))
    return x, res
