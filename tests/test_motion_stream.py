"""``dropbear_wbc.motion.stream.MotionStream``: live splicing of reference clips (real-time motion control)."""
from __future__ import annotations

import numpy as np

import sdk_test_paths  # noqa: F401  (adds source/)
from dropbear_wbc.motion.stream import Clip, MotionStream, _yaw, _yaw_quat

FPS = 50.0


def _walk(n: int, speed: float, yaw: float = 0.0, x0=(0.0, 0.0), bodies: int = 3, joints: int = 5) -> Clip:
    """A root walking straight along its heading ``yaw``; other bodies offset by 0.1 m; joints = sin waves."""
    t = np.arange(n) / FPS
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    root = np.array([x0[0], x0[1], 1.0]) + speed * t[:, None] * d
    pos = np.stack([root + np.array([0.0, 0.1 * k, 0.0]) for k in range(bodies)], axis=1)
    quat = np.broadcast_to(_yaw_quat(yaw), (n, bodies, 4)).copy()
    jp = np.sin(2 * np.pi * t[:, None] + np.arange(joints)[None])
    return Clip(jp, pos, quat)


def test_splice_aligns_position_and_heading_and_holds():
    s = MotionStream(FPS, _walk(100, 1.0), root_body_index=0, capacity_s=20.0)
    at = 60
    root_at = s.root_pos(at).copy()
    yaw_at = float(_yaw(s.body_quat_w[at, 0]))
    other = _walk(80, 0.5, yaw=1.2, x0=(5.0, -3.0))
    start, end = s.splice(other, at, blend_s=0.0)
    assert (start, end) == (60, 140)
    np.testing.assert_allclose(s.root_pos(at)[:2], root_at[:2], atol=1e-9)  # no teleport
    np.testing.assert_allclose(float(_yaw(s.body_quat_w[at, 0])), yaw_at, atol=1e-9)  # heading kept
    # the new clip walks straight along the OLD heading (x), 0.5 m/s
    np.testing.assert_allclose(s.root_pos(end - 1)[:2] - root_at[:2], [0.5 * 79 / FPS, 0.0], atol=1e-6)
    # hold: last pose, zero velocity, reported as holding
    np.testing.assert_allclose(s.joint_pos[end + 10], s.joint_pos[end - 1])
    assert np.all(s.joint_vel[end:] == 0.0) and np.all(s.body_lin_vel_w[end:] == 0.0)
    assert s.holding(end) and not s.holding(end - 1)
    assert s.dirty == (59, 141) and s.version == 2


def test_blend_is_continuous_and_velocities_follow_the_seam():
    s = MotionStream(FPS, _walk(200, 1.0), root_body_index=0, capacity_s=20.0)
    jp_before = s.joint_pos.copy()
    s.splice(_walk(100, 1.0, x0=(0.0, 0.0)), 50, blend_s=0.4)
    # no frame-to-frame jump larger than the clips' own steps across the seam
    step = np.abs(np.diff(s.joint_pos[40:80], axis=0)).max()
    own = max(np.abs(np.diff(jp_before[:100], axis=0)).max(), 1e-9)
    assert step <= 1.5 * own
    # velocities are finite differences of the written positions
    np.testing.assert_allclose(s.body_lin_vel_w[60:70, 0, 0], 1.0, atol=1e-6)


def test_interrupt_mid_clip_replaces_the_rest():
    s = MotionStream(FPS, _walk(50, 0.0), root_body_index=0, capacity_s=20.0)
    s.splice(_walk(300, 1.0), 10, blend_s=0.0)
    s.splice(_walk(40, 0.0), 120, blend_s=0.2)  # "stop" while walking
    assert s.clip_end == 160
    np.testing.assert_allclose(s.body_lin_vel_w[170], 0.0)
    assert np.all(np.isfinite(s.body_ang_vel_w))
