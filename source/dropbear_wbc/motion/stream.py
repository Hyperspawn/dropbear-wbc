"""Live reference-motion timeline: splice new clips into a running tracking reference (real-time motion control).

A tracking policy follows a reference that advances one frame per policy step. For live control (text -> motion,
teleop, a behaviour scheduler), new motion must enter that reference WHILE it plays, and may interrupt the current
motion. :class:`MotionStream` is that timeline, in numpy, in contract-NPZ layout (all articulation joints, all bodies):

* ``splice(clip, at)`` replaces everything from frame ``at`` on by ``clip``. The clip is first moved by a planar rigid
  transform (x, y, yaw) so that its root at frame 0 sits where the timeline's root is at ``at`` (height and tilt are the
  clip's own: every contract clip is settled on z = 0 ground). The first ``blend_s`` are cross-faded from the old
  timeline (joints and positions linearly, orientations by slerp, smoothstep weights); velocities are recomputed across
  the seam by finite differences.
* After the clip's last frame the timeline HOLDS that pose (zero velocity), so the tracker stands still until the next
  command.
* ``version`` increments on every splice and ``dirty`` is the frame range written (seam + clip), so a consumer (the
  simulation's motion command, a deploy runtime) copies only what changed and fills ``[clip_end, capacity)`` with the
  hold (last frame, zero velocity) itself.

Pure numpy (no Isaac); ``tests/test_motion_stream.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat(psi: float) -> np.ndarray:
    return np.array([np.cos(psi / 2.0), 0.0, 0.0, np.sin(psi / 2.0)])


def _slerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-element slerp (wxyz) with weights ``w`` broadcast over the leading axes."""
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0.0, -q1, q1)
    d = np.abs(d)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    a = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / np.where(small, 1.0, s))
    b = np.where(small, w, np.sin(w * theta) / np.where(small, 1.0, s))
    out = a * q0 + b * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _ang_vel(q: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity from consecutive orientations (T, ..., 4) -> (T, ..., 3), forward difference."""
    conj = q[:-1] * np.array([1.0, -1.0, -1.0, -1.0])
    dq = _quat_mul(q[1:], conj)  # rotation from t to t+1, world frame
    dq = np.where(dq[..., :1] < 0.0, -dq, dq)
    s = np.linalg.norm(dq[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, dq[..., :1])
    axis = dq[..., 1:] / np.where(s < 1e-9, 1.0, s)
    w = axis * angle / dt
    return np.concatenate([w, w[-1:]], axis=0)


@dataclass
class Clip:
    """One reference clip in contract layout (``MotionArrays`` fields); any leading frame count."""

    joint_pos: np.ndarray  # (T, J)
    body_pos_w: np.ndarray  # (T, B, 3)
    body_quat_w: np.ndarray  # (T, B, 4) wxyz

    @classmethod
    def from_arrays(cls, m) -> "Clip":
        return cls(np.asarray(m.joint_pos, float), np.asarray(m.body_pos_w, float), np.asarray(m.body_quat_w, float))

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


class MotionStream:
    def __init__(self, fps: float, first: Clip, root_body_index: int, capacity_s: float = 600.0):
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.root = int(root_body_index)
        self.capacity = int(round(capacity_s * self.fps))
        j, b = first.joint_pos.shape[1], first.body_pos_w.shape[1]
        self.joint_pos = np.zeros((self.capacity, j))
        self.joint_vel = np.zeros((self.capacity, j))
        self.body_pos_w = np.zeros((self.capacity, b, 3))
        self.body_quat_w = np.zeros((self.capacity, b, 4))
        self.body_lin_vel_w = np.zeros((self.capacity, b, 3))
        self.body_ang_vel_w = np.zeros((self.capacity, b, 3))
        self.version = 0
        self.dirty = (0, 0)
        self.clip_end = 0  # first HOLD frame after the last spliced clip
        self._write(first, 0, blend_n=0)

    # ------------------------------------------------------------------ splice
    def splice(self, clip: Clip, at: int, blend_s: float = 0.4, align: bool = True) -> tuple[int, int]:
        """Replace the timeline from frame ``at`` on by ``clip`` (see module doc). Returns (start, end) of the clip."""
        at = int(np.clip(at, 0, self.capacity - 2))
        if at + clip.num_frames >= self.capacity:
            raise ValueError(f"stream capacity {self.capacity} frames exceeded (splice at {at} + {clip.num_frames})")
        if align:
            clip = self._aligned(clip, at)
        self._write(clip, at, blend_n=int(round(blend_s * self.fps)))
        return at, at + clip.num_frames

    def _aligned(self, clip: Clip, at: int) -> Clip:
        """Planar rigid transform (x, y, yaw) of ``clip`` so its root at frame 0 matches the timeline root at ``at``."""
        src_q, dst_q = clip.body_quat_w[0, self.root], self.body_quat_w[at, self.root]
        dpsi = float(_yaw(dst_q) - _yaw(src_q))
        c, s = np.cos(dpsi), np.sin(dpsi)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pos = clip.body_pos_w.copy()
        o_src = clip.body_pos_w[0, self.root].copy()
        o_dst = self.body_pos_w[at, self.root].copy()
        pos[..., :2] -= o_src[:2]
        pos = pos @ rot.T
        pos[..., :2] += o_dst[:2]
        quat = _quat_mul(np.broadcast_to(_yaw_quat(dpsi), clip.body_quat_w.shape), clip.body_quat_w)
        return Clip(clip.joint_pos.copy(), pos, quat)

    def _write(self, clip: Clip, at: int, blend_n: int) -> None:
        n = clip.num_frames
        jp, bp, bq = clip.joint_pos.copy(), clip.body_pos_w.copy(), clip.body_quat_w.copy()
        nb = min(blend_n, n)
        if nb > 0:
            x = (np.arange(nb) + 1.0) / (nb + 1.0)
            w = x * x * (3.0 - 2.0 * x)  # smoothstep: 0 -> old timeline, 1 -> new clip
            jp[:nb] = (1 - w[:, None]) * self.joint_pos[at:at + nb] + w[:, None] * jp[:nb]
            bp[:nb] = (1 - w[:, None, None]) * self.body_pos_w[at:at + nb] + w[:, None, None] * bp[:nb]
            bq[:nb] = _slerp(self.body_quat_w[at:at + nb], bq[:nb], w[:, None, None])
        end = at + n
        self.joint_pos[at:end], self.body_pos_w[at:end], self.body_quat_w[at:end] = jp, bp, bq
        # hold the last pose to the end of the buffer
        self.joint_pos[end:] = jp[-1]
        self.body_pos_w[end:] = bp[-1]
        self.body_quat_w[end:] = bq[-1]
        # velocities by finite differences over [at-1, end] (the seam and the clip); zero in the hold
        lo = max(0, at - 1)
        seg = slice(lo, min(end + 1, self.capacity))
        if seg.stop - seg.start >= 2:
            self.joint_vel[seg] = np.gradient(self.joint_pos[seg], self.dt, axis=0)
            self.body_lin_vel_w[seg] = np.gradient(self.body_pos_w[seg], self.dt, axis=0)
            self.body_ang_vel_w[seg] = _ang_vel(self.body_quat_w[seg], self.dt)
        self.joint_vel[end:] = 0.0
        self.body_lin_vel_w[end:] = 0.0
        self.body_ang_vel_w[end:] = 0.0
        self.clip_end = end
        self.version += 1
        self.dirty = (lo, min(end + 1, self.capacity))  # the consumer fills [clip_end, capacity) with the hold

    # ------------------------------------------------------------------ queries
    def root_pos(self, t: int) -> np.ndarray:
        return self.body_pos_w[t, self.root]

    def holding(self, t: int) -> bool:
        """True once playback at frame ``t`` has passed the last spliced clip (the reference stands still)."""
        return t >= self.clip_end
