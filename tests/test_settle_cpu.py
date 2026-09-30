"""CPU unit tests of the settle pipeline pieces (no Isaac): CSV I/O, resampling, ground fix,
finite-difference velocities, sweep programs and numpy closure residuals.

Run: python -m pytest tests/test_settle_cpu.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ to sys.path)
from dropbear_wbc.kinematics.closures_np import ClosureTable, closure_residuals
from dropbear_wbc.kinematics.sweeps import grid2d_row, pack, single_pose, sweep1d
from dropbear_wbc.motion.rotations import quat_from_axis_angle, quat_to_matrix
from dropbear_wbc.settle.ground import SoleModel, gaussian_smooth, ground_correction
from dropbear_wbc.settle.kinematics import angular_velocity, com_positions, compose_root, linear_velocity
from dropbear_wbc.settle.motion_io import CSV_COLUMNS, MotionClip, read_motion_csv, resample, write_motion_csv


def _clip(n=31, fps=30.0):
    t = np.arange(n) / fps
    yaw = 0.5 * t
    q = np.stack([np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)], -1)
    motors = np.stack([0.1 * t * (k + 1) for k in range(22)], -1)
    pos = np.stack([t, 0.0 * t, 1.0 + 0 * t], -1)
    contact = np.stack([t < 0.5, t >= 0.5], -1)
    return MotionClip(fps, pos, q, motors, contact, {"source": "test"})


def test_csv_roundtrip(tmp_path):
    clip = _clip()
    path = tmp_path / "c.csv"
    write_motion_csv(path, clip)
    header = path.read_text().splitlines()[0].split(",")
    assert tuple(header) == CSV_COLUMNS
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["fps"] == 30.0 and len(side["contact_hint"]) == clip.num_frames
    back = read_motion_csv(path)
    np.testing.assert_allclose(back.motor_pos, clip.motor_pos, atol=1e-7)
    np.testing.assert_allclose(back.root_quat, clip.root_quat, atol=1e-7)
    np.testing.assert_array_equal(back.contact, clip.contact)


def test_csv_header_mismatch_rejected(tmp_path):
    clip = _clip()
    path = tmp_path / "c.csv"
    write_motion_csv(path, clip)
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace("LL_hip_joint", "LL_hip")
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError):
        read_motion_csv(path)


def test_resample_30_to_50_linear_and_slerp():
    clip = _clip(n=31, fps=30.0)  # duration 1.0 s
    out = resample(clip, 50.0)
    assert out.num_frames == 51 and out.fps == 50.0
    t = np.arange(51) / 50.0
    np.testing.assert_allclose(out.root_pos[:, 0], t, atol=1e-9)
    np.testing.assert_allclose(out.motor_pos[:, 3], 0.4 * t, atol=1e-9)
    yaw = 2 * np.arctan2(out.root_quat[:, 3], out.root_quat[:, 0])
    np.testing.assert_allclose(yaw, 0.5 * t, atol=1e-9)  # constant-rate yaw is exactly slerped
    assert out.contact.shape == (51, 2)


def test_resample_50_passthrough():
    clip = _clip(n=11, fps=50.0)
    out = resample(clip, 50.0)
    assert out.num_frames == 11
    np.testing.assert_allclose(out.motor_pos, clip.motor_pos, atol=1e-12)


def test_angular_and_linear_velocity():
    dt = 0.02
    t = np.arange(50) * dt
    w = np.array([0.3, -0.2, 1.1])
    q = quat_from_axis_angle(np.broadcast_to(w / np.linalg.norm(w), (50, 3)), np.linalg.norm(w) * t)
    om = angular_velocity(q[:, None, :], dt)[:, 0]
    np.testing.assert_allclose(om, np.broadcast_to(w, om.shape), atol=1e-9)
    x = np.stack([t ** 2, 3 * t, 0 * t], -1)
    v = linear_velocity(x, dt)
    np.testing.assert_allclose(v[1:-1, 0], 2 * t[1:-1], atol=1e-9)
    np.testing.assert_allclose(v[:, 1], 3.0, atol=1e-9)


def test_compose_root_and_com():
    rng = np.random.default_rng(1)
    t, b = 5, 4
    root_p = rng.normal(size=(t, 3))
    root_q = quat_from_axis_angle(rng.normal(size=(t, 3)), rng.uniform(0, 3, size=t))
    rel_p = rng.normal(size=(t, b, 3))
    rel_q = quat_from_axis_angle(rng.normal(size=(t, b, 3)), rng.uniform(0, 3, size=(t, b)))
    p, q = compose_root(root_p, root_q, rel_p, rel_q)
    r_root = quat_to_matrix(root_q)
    np.testing.assert_allclose(p, root_p[:, None] + np.einsum("tij,tbj->tbi", r_root, rel_p), atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(q), r_root[:, None] @ quat_to_matrix(rel_q), atol=1e-12)
    com_b = rng.normal(size=(b, 3))
    c = com_positions(p, q, com_b)
    np.testing.assert_allclose(c, p + np.einsum("tbij,bj->tbi", quat_to_matrix(q), com_b), atol=1e-12)


def test_ground_correction_contact_hint_and_smoothing():
    t = 100
    hl = np.full(t, 0.05)
    hr = np.full(t, 0.08)
    hr[40:60] = 0.02  # right foot lower in the middle, but the hint says the left foot is in contact
    contact = np.zeros((t, 2), bool)
    contact[:, 0] = True
    g = ground_correction({"left": hl, "right": hr}, 50.0, contact, sigma_s=0.0)
    np.testing.assert_allclose(g.dz, -0.05)
    np.testing.assert_allclose(g.contact_sole_z_after, 0.0, atol=1e-12)
    g2 = ground_correction({"left": hl, "right": hr}, 50.0, None, sigma_s=0.0)
    np.testing.assert_allclose(g2.dz[45], -0.02)  # lowest foot used when no hint
    assert g2.contact_source == "lowest_foot"
    # flight phase (no contact) is interpolated
    contact[30:40] = False
    g3 = ground_correction({"left": hl, "right": hr}, 50.0, contact, sigma_s=0.0)
    np.testing.assert_allclose(g3.dz[30:40], -0.05)


def test_gaussian_smooth_preserves_constant_and_mean():
    x = np.full(30, 2.0)
    np.testing.assert_allclose(gaussian_smooth(x, 3.0), 2.0)
    y = np.sin(np.linspace(0, 6, 200))
    assert abs(gaussian_smooth(y, 2.0).mean() - y.mean()) < 1e-2


def test_sole_model_lowest_point():
    verts = np.array([[0.0, 0.0, -0.1], [0.2, 0.0, -0.1], [0.0, 0.0, 0.1]])
    model = SoleModel({n: verts for n in ("LL_skateboard_bearing_left_2", "LL_basis_left_1",
                                          "RL_skateboard_bearing_left_2", "RL_basis_left_1")}, "x")
    names = ["world", "LL_skateboard_bearing_left_2", "LL_basis_left_1", "RL_skateboard_bearing_left_2",
             "RL_basis_left_1"]
    pos = np.zeros((1, 5, 3))
    pos[0, 1:, 2] = 1.0
    quat = np.tile([1.0, 0, 0, 0], (1, 5, 1))
    # rotate the left sole plate 90 deg about y: vertex (0.2,0,-0.1) -> z = -0.2
    quat[0, 1] = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.array(np.pi / 2))
    low = model.lowest_z(pos, quat, names)
    np.testing.assert_allclose(low["left"], [0.8], atol=1e-12)
    np.testing.assert_allclose(low["right"], [0.9], atol=1e-12)


def test_sweep_programs_small_steps_and_pack():
    deg = np.pi / 180
    p = sweep1d(3, 0.0, 30 * deg, 1 * deg, max_step=1 * deg, return_pass=True)
    tg = np.stack(p.targets)
    assert np.max(np.abs(np.diff(tg, axis=0))) <= 1 * deg + 1e-12
    rec = np.array(p.record)
    fwd = tg[rec & (np.array(p.passno) == 0), 3]
    np.testing.assert_allclose(fwd, np.linspace(0, 30 * deg, 31), atol=1e-12)
    assert (np.array(p.passno)[rec] == 1).sum() == 30
    g = grid2d_row(4, 5, -0.5, np.linspace(-0.8, 1.0, 7), max_step=2 * deg)
    tg2 = np.stack(g.targets)
    assert np.max(np.abs(np.diff(tg2, axis=0))) <= 2 * deg + 1e-12
    assert np.sum(g.record) == 7 and np.allclose(tg2[np.array(g.record), 4], -0.5)
    s = single_pose(np.full(22, 0.3), 3 * deg, "x")
    assert s.record[-1] and np.allclose(s.targets[-1], 0.3)
    batches = pack([p, g, s], num_envs=2)
    assert len(batches) == 2 and batches[0].targets.shape[1:] == (2, 22)
    assert set(batches[0].program_ids.tolist() + batches[1].program_ids.tolist()) == {0, 1, 2, -1}


def test_closure_residuals_zero_and_known_gap():
    table = ClosureTable(
        names=np.array(["fix", "rev", "sph"]), types=np.array(["PhysicsFixedJoint", "PhysicsRevoluteJoint",
                                                               "PhysicsSphericalJoint"]),
        axes=np.array(["", "Z", ""]), body0=np.array([0, 0, 1]), body1=np.array([1, 1, 0]),
        local_pos0=np.array([[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]), local_pos1=np.array([[0.0, 0, 0]] * 3),
        local_rot0=np.tile([1.0, 0, 0, 0], (3, 1)), local_rot1=np.tile([1.0, 0, 0, 0], (3, 1)),
    )
    pos = np.array([[0.0, 0, 0], [0.1, 0, 0]])
    quat = np.tile([1.0, 0, 0, 0], (2, 1))
    gaps, ang = closure_residuals(pos, quat, table)
    np.testing.assert_allclose(gaps, [0.0, 0.1, 0.1], atol=1e-12)
    np.testing.assert_allclose(ang, 0.0, atol=1e-7)
    # rotate body 1 about z (the revolute's free axis): fixed joint sees the angle, revolute does not
    quat[1] = quat_from_axis_angle(np.array([0, 0, 1.0]), np.array(0.3))
    _, ang = closure_residuals(pos, quat, table)
    np.testing.assert_allclose(ang, [0.3, 0.0, 0.0], atol=1e-7)


def test_sole_plane_recovers_tilted_bottom_face():
    from dropbear_wbc.kinematics.calib_build import sole_plane

    rng = np.random.default_rng(0)
    # a 30 x 10 x 3 cm plate (dense surface points) with raised ribs on top, pitched 3.3 deg toe-up
    x, y = np.meshgrid(np.linspace(-0.05, 0.25, 61), np.linspace(-0.05, 0.05, 21), indexing="ij")
    bottom = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], -1)
    top = bottom + [0, 0, 0.03]
    ribs = top[rng.random(len(top)) < 0.2] + [0, 0, 0.02]
    pts = np.concatenate([bottom, top, ribs])
    ang = np.radians(3.3)
    r = np.array([[np.cos(ang), 0, -np.sin(ang)], [0, 1, 0], [np.sin(ang), 0, np.cos(ang)]])  # toe (x+) up
    n, pt = sole_plane(pts @ r.T + [0.0, 0.0, 0.13])
    np.testing.assert_allclose(n, r @ [0, 0, 1], atol=1e-6)
    assert np.degrees(np.arctan2(-n[0], n[2])) == pytest.approx(3.3, abs=1e-3)


def test_point_pivot_recovers_fixed_centre():
    from dropbear_wbc.kinematics.calib_build import point_pivot
    from dropbear_wbc.kinematics.rigid import axis_angle_matrix

    c = np.array([0.1, -0.02, -0.5])
    a0 = np.array([0.05, 0.0, -0.9])
    ang = np.linspace(0, 0.8, 30)
    r = axis_angle_matrix(np.broadcast_to([0.0, 1.0, 0.0], (30, 3)), ang)
    pts = c + np.einsum("nij,j->ni", r, a0 - c)
    est, rms = point_pivot(r, pts, a0)
    assert rms < 1e-12
    # recovered up to a shift along the rotation axis (y)
    np.testing.assert_allclose(est[[0, 2]], c[[0, 2]], atol=1e-9)
