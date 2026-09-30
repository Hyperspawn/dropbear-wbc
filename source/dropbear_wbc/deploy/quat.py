"""Small numpy quaternion helpers. Convention: ``wxyz``, Hamilton product, active rotations.

``quat_rotate(q, v)`` maps a vector from the frame described by ``q`` into its
parent frame (body -> world for a body orientation), matching Isaac Lab's
``isaaclab.utils.math`` functions of the same names.
"""
from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (wxyz)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by ``q``."""
    q = np.asarray(q, np.float64)
    u, w = q[..., 1:], q[..., :1]
    v = np.asarray(v, np.float64)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of ``q`` (world -> body for a body orientation)."""
    return rotate(conj(q), v)


def matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix of a unit quaternion."""
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw(q: np.ndarray) -> float:
    """Heading angle [rad] about world z."""
    w, x, y, z = np.asarray(q, np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def from_yaw(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Heading-only part of ``q``."""
    return from_yaw(yaw(q))


def subtract_frame_transforms(p01: np.ndarray, q01: np.ndarray, p02: np.ndarray, q02: np.ndarray):
    """Pose of frame 2 expressed in frame 1: ``(R1^T (p2 - p1), q1^-1 q2)`` (Isaac Lab semantics)."""
    q1_inv = conj(q01)
    return rotate(q1_inv, np.asarray(p02, np.float64) - np.asarray(p01, np.float64)), mul(q1_inv, q02)


def rpy(q: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw [rad] (intrinsic ZYX)."""
    w, x, y, z = np.asarray(q, np.float64)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.array([roll, pitch, yaw(q)])


def tilt_angle(q: np.ndarray) -> float:
    """Angle [rad] between the body z axis and world z (0 = upright)."""
    zb = rotate(q, np.array([0.0, 0.0, 1.0]))
    return float(np.arccos(np.clip(zb[2], -1.0, 1.0)))
